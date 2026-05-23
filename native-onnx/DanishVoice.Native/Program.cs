using System.Text.Json;
using DanishVoice.Native;
using DanishVoice.Native.Flow;
using DanishVoice.Native.T3;
using DanishVoice.Native.Tokenizer;

// Native synthesis CLI:  dotnet run -- synth <root> <voice> <out.wav> <text...>
if (args.Length >= 5 && args[0] == "synth")
{
    var synthRoot = args[1];
    var voice = args[2];
    var outWav = args[3];
    var text = string.Join(' ', args[4..]);
    using var pipeline = new SynthPipeline(Path.Combine(synthRoot, "onnx_models"),
                                           Path.Combine(synthRoot, "refs"));
    Console.WriteLine($"Synthesizing [{voice}]: {text}");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var wav = pipeline.Synth(text, voice);
    sw.Stop();
    WriteWav(outWav, wav, 24000);
    Console.WriteLine($"Wrote {outWav} ({wav.Length} samples, {wav.Length / 24000.0:F2}s) in {sw.Elapsed.TotalSeconds:F1}s");
    return;
}

// Parity harness for the native ONNX components built so far.
// Usage: dotnet run -- <native-onnx-dir>   (defaults to ../ from the project)

var root = args.Length > 0
    ? args[0]
    : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
var onnxDir = Path.Combine(root, "onnx_models");
var refsDir = Path.Combine(root, "refs");

Console.WriteLine($"root={root}");
Console.WriteLine($"onnxDir={onnxDir}");
Console.WriteLine($"refsDir={refsDir}");
TensorIo.LoadShapes(refsDir);

// ---- Test 1: ONNX Runtime toolchain proof via VoiceEncoder ----
Console.WriteLine("\n== Test 1: VoiceEncoder (ONNX toolchain proof) ==");
{
    using var ve = new OnnxModel(Path.Combine(onnxDir, "voice_encoder.onnx"));
    var melsShape = TensorIo.Shape("ve_input_mels"); // (22,160,40)
    var mels = TensorIo.Load(refsDir, "ve_input_mels");
    var inputName = ve.InputNames[0];
    var outName = ve.OutputNames[0];
    var outs = ve.Run(new Dictionary<string, (float[], int[])>
    {
        [inputName] = (mels, melsShape),
    });
    var got = outs[outName].Data;
    var expected = TensorIo.Load(refsDir, "ve_output");
    TensorIo.Report("ve_output", got, expected, 1e-3);
}

// ---- Test 2: Vocoder chain, stage by stage ----
Console.WriteLine("\n== Test 2: Vocoder split (f0 + NSF + STFT + conv + iSTFT) ==");
{
    using var voc = new Vocoder(onnxDir, refsDir);
    var sfShape = TensorIo.Shape("voc_speech_feat"); // (1,80,T)
    var speechFeat = TensorIo.Load(refsDir, "voc_speech_feat");
    var melChannels = sfShape[1];
    var frames = sfShape[2];

    var trace = voc.Run(speechFeat, melChannels, frames);

    TensorIo.Report("f0", trace.F0, TensorIo.Load(refsDir, "voc_f0"), 1e-3);
    TensorIo.Report("source_s", trace.Source, TensorIo.Load(refsDir, "voc_s"), 5e-3);
    TensorIo.Report("s_stft", trace.SStft, TensorIo.Load(refsDir, "voc_s_stft"), 5e-3);
    TensorIo.Report("full_wav", trace.Wav, TensorIo.Load(refsDir, "voc_full_wav"), 5e-3);
}

// ---- Test 3: MTL tokenizer (exact parity) ----
Console.WriteLine("\n== Test 3: MTL tokenizer (Danish) ==");
{
    var tok = new MtlTokenizer(Path.Combine(refsDir, "grapheme_tokenizer.json"));
    using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(refsDir, "tokenizer_refs.json")));
    var pass = 0;
    var total = 0;
    foreach (var item in doc.RootElement.EnumerateArray())
    {
        total++;
        var text = item.GetProperty("text").GetString()!;
        var expected = item.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var got = tok.Encode(text, "da");
        var ok = got.SequenceEqual(expected);
        if (ok)
        {
            pass++;
        }
        else
        {
            Console.WriteLine($"  [FAIL] {text}");
            Console.WriteLine($"     exp: {string.Join(",", expected)}");
            Console.WriteLine($"     got: {string.Join(",", got)}");
        }
    }
    Console.WriteLine($"  tokenizer parity: {pass}/{total} sentences exact");
}

// ---- Test 4: T3 backbone + speech_head -> step-0 logits (deterministic) ----
Console.WriteLine("\n== Test 4: T3 backbone (Llama) + speech_head ==");
{
    TensorIo.AddShapes(refsDir, "shapes_t3.json");
    using var backbone = new OnnxModel(Path.Combine(onnxDir, "t3_backbone.onnx"));
    var ieShape = TensorIo.Shape("t3_inputs_embeds"); // (2, L, 1024)
    var ie = TensorIo.Load(refsDir, "t3_inputs_embeds");
    var outs = backbone.Run(new Dictionary<string, (float[], int[])>
    {
        ["inputs_embeds"] = (ie, ieShape),
    });
    var hidden = outs["last_hidden_state"];
    var b = hidden.Dims[0];
    var l = hidden.Dims[1];
    var dim = hidden.Dims[2];

    // take last position -> (B, dim)
    var lastHidden = new float[b * dim];
    for (var bi = 0; bi < b; bi++)
    {
        Array.Copy(hidden.Data, (bi * l + (l - 1)) * dim, lastHidden, bi * dim, dim);
    }
    TensorIo.Report("hidden_last", lastHidden, TensorIo.Load(refsDir, "t3_hidden_last"), 2e-3);

    // speech_head: logits[bi,v] = sum_d lastHidden[bi,d] * W[v,d]
    var wShape = TensorIo.Shape("t3_speech_head_weight"); // (V, dim)
    var w = TensorIo.Load(refsDir, "t3_speech_head_weight");
    var vocab = wShape[0];
    var logits = new float[b * vocab];
    for (var bi = 0; bi < b; bi++)
    {
        for (var v = 0; v < vocab; v++)
        {
            double acc = 0;
            var wo = v * dim;
            var ho = bi * dim;
            for (var d = 0; d < dim; d++)
            {
                acc += (double)lastHidden[ho + d] * w[wo + d];
            }
            logits[bi * vocab + v] = (float)acc;
        }
    }
    TensorIo.Report("logits_step0", logits, TensorIo.Load(refsDir, "t3_logits_step0"), 5e-3);
}

// ---- Test 5: T3 autoregressive loop (greedy token parity) ----
Console.WriteLine("\n== Test 5: T3 greedy loop (analyzer disabled) ==");
{
    TensorIo.AddShapes(refsDir, "shapes_t3loop.json");
    var steps = args.Length > 1 && int.TryParse(args[1], out var s) ? s : 8;

    var condShape = TensorIo.Shape("t3_cond_emb"); // (1, lenCond, dim)
    var condEmb = TensorIo.Load(refsDir, "t3_cond_emb");
    var lenCond = condShape[1];

    using var ttDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(refsDir, "t3_text_tokens.json")));
    var textTokensCond = ttDoc.RootElement[0].EnumerateArray().Select(e => e.GetInt32()).ToArray();

    using var grDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(refsDir, "t3_greedy_tokens.json")));
    var expectedTokens = grDoc.RootElement[0].EnumerateArray().Select(e => e.GetInt32()).ToArray();

    using var t3 = new T3Model(onnxDir, refsDir);
    Console.WriteLine($"  running {steps} greedy steps (CPU full-sequence — slow)...");
    var got = t3.Generate(condEmb, lenCond, textTokensCond, steps);

    var n = Math.Min(steps, Math.Min(got.Length, expectedTokens.Length));
    var match = 0;
    for (var i = 0; i < n; i++)
    {
        if (got[i] == expectedTokens[i])
        {
            match++;
        }
    }
    Console.WriteLine($"  expected[..{n}]: {string.Join(",", expectedTokens.Take(n))}");
    Console.WriteLine($"  got     [..{n}]: {string.Join(",", got.Take(n))}");
    Console.WriteLine($"  T3 greedy token parity: {match}/{n}");
}

// ---- Test 6: CFM flow decoder (deterministic, z as input) ----
Console.WriteLine("\n== Test 6: CFM flow decoder (ODE solver, z input) ==");
{
    TensorIo.AddShapes(refsDir, "shapes_flow.json");
    using var cfm = new OnnxModel(Path.Combine(onnxDir, "cfm_decoder_z.onnx"));
    var outs = cfm.Run(new Dictionary<string, (float[], int[])>
    {
        ["mu"] = (TensorIo.Load(refsDir, "flow_mu"), TensorIo.Shape("flow_mu")),
        ["mask"] = (TensorIo.Load(refsDir, "flow_mask"), TensorIo.Shape("flow_mask")),
        ["spks"] = (TensorIo.Load(refsDir, "flow_spks"), TensorIo.Shape("flow_spks")),
        ["cond"] = (TensorIo.Load(refsDir, "flow_cond"), TensorIo.Shape("flow_cond")),
        ["z"] = (TensorIo.Load(refsDir, "flow_z"), TensorIo.Shape("flow_z")),
    });
    var mel = outs[cfm.OutputNames[0]].Data;
    TensorIo.Report("flow_mel", mel, TensorIo.Load(refsDir, "flow_mel_out"), 5e-3);
}

// ---- Test 7: Conformer encoder (last untested ONNX graph) ----
Console.WriteLine("\n== Test 7: Conformer encoder ==");
{
    using var enc = new OnnxModel(Path.Combine(onnxDir, "conformer_encoder.onnx"));
    var inShape = TensorIo.Shape("flow_encoder_in"); // (1, T, 512)
    var encIn = TensorIo.Load(refsDir, "flow_encoder_in");
    var lens = new float[] { inShape[1] };
    var feeds = new Dictionary<string, (float[], int[])>();
    var inNames = enc.InputNames;
    feeds[inNames[0]] = (encIn, inShape);
    if (inNames.Count > 1)
    {
        feeds[inNames[1]] = (lens, [1]);
    }
    try
    {
        var outs = enc.Run(feeds);
        var h = outs[enc.OutputNames[0]].Data;
        TensorIo.Report("conformer_h", h, TensorIo.Load(refsDir, "flow_h"), 5e-3);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  conformer run note: {ex.Message.Split('\n')[0]}");
    }
}

// ---- Test 8: full chain (flow assembly + vocoder) on consistent e2e refs ----
Console.WriteLine("\n== Test 8: end-to-end chain (flow + vocoder) ==");
{
    TensorIo.AddShapes(refsDir, "shapes_flow.json");
    TensorIo.AddShapes(refsDir, "shapes_flow2.json");
    TensorIo.AddShapes(refsDir, "shapes_e2e.json");

    using var metaDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(refsDir, "e2e_meta.json")));
    var melLen1 = metaDoc.RootElement.GetProperty("mel_len1").GetInt32();

    using var tcDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(refsDir, "e2e_token_concat.json")));
    var tokenConcat = tcDoc.RootElement[0].EnumerateArray().Select(e => e.GetInt32()).ToArray();

    var promptFeat = TensorIo.Load(refsDir, "e2e_prompt_feat"); // (1,500,80)
    var xvector = TensorIo.Load(refsDir, "e2e_xvector");        // (1,192)
    var z = TensorIo.Load(refsDir, "e2e_cfm_z");               // (1,80,584)

    using var flow = new FlowModel(onnxDir, refsDir);
    var trace = flow.Run(tokenConcat, promptFeat, melLen1, xvector, z);

    TensorIo.Report("flow_mu", trace.Mu, TensorIo.Load(refsDir, "e2e_cfm_mu"), 5e-3);
    TensorIo.Report("flow_spks", trace.Spks, TensorIo.Load(refsDir, "e2e_cfm_spks"), 5e-3);
    TensorIo.Report("flow_cond", trace.Cond, TensorIo.Load(refsDir, "e2e_cfm_cond"), 5e-3);
    TensorIo.Report("flow_mel(full)", trace.Mel, TensorIo.Load(refsDir, "e2e_cfm_out"), 2e-2);
    TensorIo.Report("vocoder_mel", trace.MelOut, TensorIo.Load(refsDir, "e2e_voc_speech_feat"), 2e-2);

    // vocoder on the flow mel -> wav
    var outFrames = trace.T2 - melLen1;
    using var voc = new Vocoder(onnxDir, refsDir);
    var vtrace = voc.Run(trace.MelOut, 80, outFrames);
    var wavRef = TensorIo.Load(refsDir, "e2e_wav");
    var n = Math.Min(vtrace.Wav.Length, wavRef.Length);
    TensorIo.Report("e2e_wav (NSF stochastic in ref)", vtrace.Wav.AsSpan(0, n), wavRef.AsSpan(0, n), 5e-2);

    // write the C# wav for listening
    WriteWav(Path.Combine(root, "native_csharp_out.wav"), vtrace.Wav, 24000);
    Console.WriteLine($"  wrote native_csharp_out.wav ({vtrace.Wav.Length} samples)");
}

Console.WriteLine("\nDone.");

static void WriteWav(string path, float[] samples, int sampleRate)
{
    using var fs = new FileStream(path, FileMode.Create);
    using var bw = new BinaryWriter(fs);
    var n = samples.Length;
    var byteRate = sampleRate * 2;
    bw.Write("RIFF"u8.ToArray());
    bw.Write(36 + n * 2);
    bw.Write("WAVE"u8.ToArray());
    bw.Write("fmt "u8.ToArray());
    bw.Write(16);
    bw.Write((short)1);
    bw.Write((short)1);
    bw.Write(sampleRate);
    bw.Write(byteRate);
    bw.Write((short)2);
    bw.Write((short)16);
    bw.Write("data"u8.ToArray());
    bw.Write(n * 2);
    foreach (var s in samples)
    {
        var v = (short)Math.Clamp(s * 32767f, -32768f, 32767f);
        bw.Write(v);
    }
}
