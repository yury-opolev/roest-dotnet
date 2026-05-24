using System.Text.Json;

namespace DanishVoice.Native.T3;

/// <summary>
/// T3 autoregressive decoder in C#: builds the prefill inputs_embeds (offline
/// cond_emb + text/speech embeddings, batch-2 for CFG), then runs the sampling
/// loop against the exported Llama backbone, applying CFG combine + HF logits
/// processors. Full-sequence (no KV cache) for correctness.
///
/// NOTE: the multilingual alignment-stream analyzer is not yet ported, so this
/// matches a Python reference generated with the analyzer disabled.
/// </summary>
internal sealed class T3Model : IDisposable
{
    private readonly OnnxModel? backbone;
    private readonly OnnxModel? prefill;
    private readonly OnnxModel? decode;
    private readonly bool useKvCache;
    private readonly int nLayers;
    private readonly EmbeddingTables tables;
    private readonly float[] speechHeadWeight; // (vocab, dim)
    private readonly int vocab;
    private readonly int dim;
    private readonly int startSpeechToken;
    private readonly int stopSpeechToken;

    public T3Model(string onnxDir, string refsDir, bool useCuda = false)
    {
        this.tables = new EmbeddingTables(refsDir);
        this.dim = this.tables.Dim;
        var whapes = TensorIo.Shape("t3_speech_head_weight");
        this.vocab = whapes[0];
        this.speechHeadWeight = TensorIo.Load(refsDir, "t3_speech_head_weight");

        using var cfgDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(refsDir, "t3_config.json")));
        this.startSpeechToken = cfgDoc.RootElement.GetProperty("start_speech_token").GetInt32();
        this.stopSpeechToken = cfgDoc.RootElement.GetProperty("stop_speech_token").GetInt32();

        // Prefer the KV-cache graphs (constant-cost decode) when present; fall
        // back to the O(n^2) full-sequence backbone otherwise.
        var prefillPath = Path.Combine(onnxDir, "t3_prefill.onnx");
        var decodePath = Path.Combine(onnxDir, "t3_decode.onnx");
        if (File.Exists(prefillPath) && File.Exists(decodePath))
        {
            this.prefill = new OnnxModel(prefillPath, useCuda);
            this.decode = new OnnxModel(decodePath, useCuda);
            this.nLayers = (this.prefill.OutputNames.Count - 1) / 2; // hidden + (k,v)*L
            this.useKvCache = true;
        }
        else
        {
            this.backbone = new OnnxModel(Path.Combine(onnxDir, "t3_backbone.onnx"), useCuda);
        }
    }

    public int[] Generate(
        float[] condEmb, int lenCond,
        int[] textTokensCond,
        int maxNewTokens,
        float temperature = 0.8f, float topP = 0.95f, float minP = 0.05f,
        float repetitionPenalty = 2.0f, float cfgWeight = 0.5f,
        bool greedy = true)
    {
        var lenText = textTokensCond.Length;
        var prefixLen = lenCond + lenText + 2; // cond + text + initial-speech + bos

        // build initial inputs_embeds (2, prefixLen, dim) row-major flat
        var embeds = new List<float[]>(); // each entry is one (2*dim) position block? simpler: build per-batch lists
        // We store as a growing list of positions; each position holds 2 rows x dim.
        var seq = new List<float[]>(prefixLen); // each item length 2*dim (row0 then row1)

        // cond positions (same for both rows)
        for (var p = 0; p < lenCond; p++)
        {
            var blk = new float[2 * this.dim];
            Array.Copy(condEmb, p * this.dim, blk, 0, this.dim);          // row0
            Array.Copy(condEmb, p * this.dim, blk, this.dim, this.dim);   // row1
            seq.Add(blk);
        }
        // text positions: row0 real token emb, row1 zeroed-token (pos only)
        for (var p = 0; p < lenText; p++)
        {
            var blk = new float[2 * this.dim];
            this.tables.TextEmbedding(textTokensCond[p], p, blk.AsSpan(0, this.dim), zeroToken: false);
            this.tables.TextEmbedding(textTokensCond[p], p, blk.AsSpan(this.dim, this.dim), zeroToken: true);
            seq.Add(blk);
        }
        // initial speech token (start), pos 0, both rows
        {
            var blk = new float[2 * this.dim];
            this.tables.SpeechEmbedding(this.startSpeechToken, 0, blk.AsSpan(0, this.dim));
            this.tables.SpeechEmbedding(this.startSpeechToken, 0, blk.AsSpan(this.dim, this.dim));
            seq.Add(blk);
        }
        // appended bos token (start), pos 0, both rows
        {
            var blk = new float[2 * this.dim];
            this.tables.SpeechEmbedding(this.startSpeechToken, 0, blk.AsSpan(0, this.dim));
            this.tables.SpeechEmbedding(this.startSpeechToken, 0, blk.AsSpan(this.dim, this.dim));
            seq.Add(blk);
        }

        if (this.useKvCache)
        {
            return this.GenerateKv(seq, maxNewTokens, temperature, topP, minP, repetitionPenalty, cfgWeight);
        }

        var generated = new List<int> { this.startSpeechToken };
        var outTokens = new List<int>();

        for (var step = 0; step < maxNewTokens; step++)
        {
            var l = seq.Count;
            var flat = new float[2 * l * this.dim];
            // ONNX layout (B=2, L, dim): row-major [b][pos][d]
            for (var pos = 0; pos < l; pos++)
            {
                var blk = seq[pos];
                Array.Copy(blk, 0, flat, (0 * l + pos) * this.dim, this.dim);
                Array.Copy(blk, this.dim, flat, (1 * l + pos) * this.dim, this.dim);
            }

            var outs = this.backbone!.Run(new Dictionary<string, (float[], int[])>
            {
                ["inputs_embeds"] = (flat, [2, l, this.dim]),
            });
            var hidden = outs["last_hidden_state"].Data; // (2, l, dim)

            // last position hidden for both rows
            var hCond = new float[this.dim];
            var hUncond = new float[this.dim];
            Array.Copy(hidden, (0 * l + (l - 1)) * this.dim, hCond, 0, this.dim);
            Array.Copy(hidden, (1 * l + (l - 1)) * this.dim, hUncond, 0, this.dim);

            var logitsCond = this.SpeechHead(hCond);
            var logitsUncond = this.SpeechHead(hUncond);

            // CFG combine
            var logits = new float[this.vocab];
            for (var v = 0; v < this.vocab; v++)
            {
                logits[v] = logitsCond[v] + cfgWeight * (logitsCond[v] - logitsUncond[v]);
            }

            LogitsProcessors.RepetitionPenalty(logits, generated, repetitionPenalty);
            LogitsProcessors.Temperature(logits, temperature);
            LogitsProcessors.MinP(logits, minP);
            LogitsProcessors.TopP(logits, topP);

            var next = LogitsProcessors.ArgMax(logits); // greedy reference path

            outTokens.Add(next);
            generated.Add(next);
            if (next == this.stopSpeechToken)
            {
                break;
            }

            var blk2 = new float[2 * this.dim];
            this.tables.SpeechEmbedding(next, step + 1, blk2.AsSpan(0, this.dim));
            this.tables.SpeechEmbedding(next, step + 1, blk2.AsSpan(this.dim, this.dim));
            seq.Add(blk2);
        }

        return [.. outTokens];
    }

    // KV-cached decode: prefill once, then 1 token per step (constant cost).
    private int[] GenerateKv(
        List<float[]> seq, int maxNewTokens,
        float temperature, float topP, float minP, float repetitionPenalty, float cfgWeight)
    {
        var prefixLen = seq.Count;
        var flat = new float[2 * prefixLen * this.dim];
        for (var pos = 0; pos < prefixLen; pos++)
        {
            Array.Copy(seq[pos], 0, flat, (0 * prefixLen + pos) * this.dim, this.dim);
            Array.Copy(seq[pos], this.dim, flat, (1 * prefixLen + pos) * this.dim, this.dim);
        }

        var pf = this.prefill!.Run(new Dictionary<string, (float[], int[])>
        {
            ["inputs_embeds"] = (flat, [2, prefixLen, this.dim]),
        });
        var hidden = pf["hidden"].Data; // (2, prefixLen, dim)

        var cache = new Dictionary<string, (float[], int[])>(2 * this.nLayers);
        for (var i = 0; i < this.nLayers; i++)
        {
            cache[$"past_k{i}"] = pf[$"present_k{i}"];
            cache[$"past_v{i}"] = pf[$"present_v{i}"];
        }

        var generated = new List<int> { this.startSpeechToken };
        var outTokens = new List<int>();

        var tok = this.SampleNext(hidden, prefixLen, generated, temperature, topP, minP, repetitionPenalty, cfgWeight);
        outTokens.Add(tok);
        generated.Add(tok);

        for (var step = 1; step < maxNewTokens && tok != this.stopSpeechToken; step++)
        {
            var emb = new float[2 * this.dim];
            this.tables.SpeechEmbedding(tok, step, emb.AsSpan(0, this.dim));
            this.tables.SpeechEmbedding(tok, step, emb.AsSpan(this.dim, this.dim));

            var inputs = new Dictionary<string, (float[], int[])>(cache)
            {
                ["inputs_embeds"] = (emb, [2, 1, this.dim]),
            };
            var dec = this.decode!.Run(inputs);
            for (var i = 0; i < this.nLayers; i++)
            {
                cache[$"past_k{i}"] = dec[$"present_k{i}"];
                cache[$"past_v{i}"] = dec[$"present_v{i}"];
            }

            tok = this.SampleNext(dec["hidden"].Data, 1, generated, temperature, topP, minP, repetitionPenalty, cfgWeight);
            outTokens.Add(tok);
            generated.Add(tok);
        }

        // mirror the full-sequence path, which appends the stop token then stops
        return [.. outTokens];
    }

    // Computes CFG-combined, processed logits from the last position of a
    // (2, len, dim) hidden tensor and returns the greedy next token.
    private int SampleNext(
        float[] hidden, int len, List<int> generated,
        float temperature, float topP, float minP, float repetitionPenalty, float cfgWeight)
    {
        var hCond = new float[this.dim];
        var hUncond = new float[this.dim];
        Array.Copy(hidden, (0 * len + (len - 1)) * this.dim, hCond, 0, this.dim);
        Array.Copy(hidden, (1 * len + (len - 1)) * this.dim, hUncond, 0, this.dim);

        var logitsCond = this.SpeechHead(hCond);
        var logitsUncond = this.SpeechHead(hUncond);
        var logits = new float[this.vocab];
        for (var v = 0; v < this.vocab; v++)
        {
            logits[v] = logitsCond[v] + cfgWeight * (logitsCond[v] - logitsUncond[v]);
        }

        LogitsProcessors.RepetitionPenalty(logits, generated, repetitionPenalty);
        LogitsProcessors.Temperature(logits, temperature);
        LogitsProcessors.MinP(logits, minP);
        LogitsProcessors.TopP(logits, topP);
        return LogitsProcessors.ArgMax(logits);
    }

    private float[] SpeechHead(float[] hidden)
    {
        var logits = new float[this.vocab];
        for (var v = 0; v < this.vocab; v++)
        {
            double acc = 0;
            var wo = v * this.dim;
            for (var d = 0; d < this.dim; d++)
            {
                acc += (double)hidden[d] * this.speechHeadWeight[wo + d];
            }
            logits[v] = (float)acc;
        }
        return logits;
    }

    public void Dispose()
    {
        this.backbone?.Dispose();
        this.prefill?.Dispose();
        this.decode?.Dispose();
    }
}
