namespace DanishVoice.Native.Flow;

/// <summary>
/// S3Gen flow path in C#: input_embedding(tokens) -> conformer encoder (ONNX)
/// -> encoder_proj (linear) = mu; speaker affine = spks; cond = prompt_feat in
/// the first mel_len1 frames; then the CFM ODE solver (ONNX, z supplied) -> mel,
/// sliced to drop the prompt region. Mirrors CausalMaskedDiffWithXvec.inference.
/// </summary>
internal sealed class FlowModel : IDisposable
{
    private readonly OnnxModel conformer;
    private readonly OnnxModel cfm;
    private readonly float[] inputEmbedding; // (vocab, encDim)
    private readonly int encDim;
    private readonly float[] projW;          // (80, encDim)
    private readonly float[] projB;          // (80,)
    private readonly float[] affineW;        // (80, xvecDim)
    private readonly float[] affineB;        // (80,)
    private readonly int xvecDim;
    private const int MelDim = 80;

    public FlowModel(string onnxDir, string refsDir, bool useCuda = false)
    {
        this.conformer = new OnnxModel(Path.Combine(onnxDir, "conformer_encoder_dyn.onnx"), useCuda);
        this.cfm = new OnnxModel(Path.Combine(onnxDir, "cfm_decoder_z.onnx"), useCuda);

        this.encDim = TensorIo.Shape("flow_input_embedding_weight")[1];
        this.inputEmbedding = TensorIo.Load(refsDir, "flow_input_embedding_weight");
        this.projW = TensorIo.Load(refsDir, "flow_encoder_proj_weight");
        this.projB = TensorIo.Load(refsDir, "flow_encoder_proj_bias");
        this.affineW = TensorIo.Load(refsDir, "flow_spk_affine_weight");
        this.affineB = TensorIo.Load(refsDir, "flow_spk_affine_bias");
        this.xvecDim = TensorIo.Shape("flow_spk_affine_weight")[1];
    }

    public sealed class Trace
    {
        public float[] Mu = [];     // (1, 80, T2)
        public float[] Spks = [];   // (1, 80)
        public float[] Cond = [];   // (1, 80, T2)
        public float[] Mel = [];    // (1, 80, T2)
        public float[] MelOut = []; // (1, 80, T2 - mel_len1)
        public int T2;
    }

    /// <summary>
    /// tokenConcat: prompt_token ++ speech_tokens. promptFeat: (1, melLen1, 80).
    /// xvector: (xvecDim,). z: (1, 80, 2*len) noise. Returns mel for the vocoder.
    /// </summary>
    public Trace Run(int[] tokenConcat, float[] promptFeat, int melLen1, float[] xvector, float[] z)
    {
        var t = tokenConcat.Length;

        // input_embedding (mask is all-ones at full length)
        var emb = new float[t * this.encDim];
        for (var i = 0; i < t; i++)
        {
            Array.Copy(this.inputEmbedding, tokenConcat[i] * this.encDim, emb, i * this.encDim, this.encDim);
        }

        // conformer encoder -> h (1, T2, encDim)
        var encOut = this.conformer.Run(
            new Dictionary<string, (float[], int[])> { ["xs"] = (emb, [1, t, this.encDim]) },
            new Dictionary<string, (long[], int[])> { ["xs_lens"] = ([t], [1]) });
        var h = encOut["h"].Data;
        var t2 = encOut["h"].Dims[1];

        // encoder_proj (linear encDim->80) then transpose to (1, 80, T2) = mu
        var mu = new float[MelDim * t2];
        for (var f = 0; f < t2; f++)
        {
            var ho = f * this.encDim;
            for (var m = 0; m < MelDim; m++)
            {
                double acc = this.projB[m];
                var wo = m * this.encDim;
                for (var d = 0; d < this.encDim; d++)
                {
                    acc += (double)h[ho + d] * this.projW[wo + d];
                }
                mu[m * t2 + f] = (float)acc;
            }
        }

        // speaker embedding: L2-normalize xvector, then affine (xvecDim->80)
        var spks = this.SpeakerEmbedding(xvector);

        // cond: zeros (80, T2), first melLen1 frames = prompt_feat^T
        var cond = new float[MelDim * t2];
        for (var f = 0; f < melLen1 && f < t2; f++)
        {
            for (var m = 0; m < MelDim; m++)
            {
                cond[m * t2 + f] = promptFeat[f * MelDim + m];
            }
        }

        var mask = new float[t2];
        Array.Fill(mask, 1f);

        // CFM ODE solver (ONNX) with supplied z
        var cfmOut = this.cfm.Run(new Dictionary<string, (float[], int[])>
        {
            ["mu"] = (mu, [1, MelDim, t2]),
            ["mask"] = (mask, [1, 1, t2]),
            ["spks"] = (spks, [1, MelDim]),
            ["cond"] = (cond, [1, MelDim, t2]),
            ["z"] = (z, [1, MelDim, t2]),
        });
        var mel = cfmOut[this.cfm.OutputNames[0]].Data;

        // slice off the prompt region: mel[:, :, melLen1:]
        var outFrames = t2 - melLen1;
        var melOut = new float[MelDim * outFrames];
        for (var m = 0; m < MelDim; m++)
        {
            for (var f = 0; f < outFrames; f++)
            {
                melOut[m * outFrames + f] = mel[m * t2 + (melLen1 + f)];
            }
        }

        return new Trace { Mu = mu, Spks = spks, Cond = cond, Mel = mel, MelOut = melOut, T2 = t2 };
    }

    private float[] SpeakerEmbedding(float[] xvector)
    {
        double norm = 0;
        for (var i = 0; i < this.xvecDim; i++)
        {
            norm += (double)xvector[i] * xvector[i];
        }
        norm = Math.Sqrt(norm);
        var normed = new float[this.xvecDim];
        for (var i = 0; i < this.xvecDim; i++)
        {
            normed[i] = (float)(xvector[i] / norm);
        }
        var spks = new float[MelDim];
        for (var m = 0; m < MelDim; m++)
        {
            double acc = this.affineB[m];
            var wo = m * this.xvecDim;
            for (var i = 0; i < this.xvecDim; i++)
            {
                acc += (double)normed[i] * this.affineW[wo + i];
            }
            spks[m] = (float)acc;
        }
        return spks;
    }

    public void Dispose()
    {
        this.conformer.Dispose();
        this.cfm.Dispose();
    }
}
