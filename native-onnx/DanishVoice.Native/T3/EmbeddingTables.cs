namespace DanishVoice.Native.T3;

/// <summary>
/// T3 embedding tables (text/speech token embeddings + learned position
/// embeddings), loaded from the dumped weight tensors. LearnedPositionEmbeddings
/// is just a lookup: position p -> table row p.
/// </summary>
internal sealed class EmbeddingTables
{
    private readonly float[] textEmb;       // (textVocab, dim)
    private readonly float[] speechEmb;     // (speechVocab, dim)
    private readonly float[] textPos;       // (maxText, dim)
    private readonly float[] speechPos;     // (maxSpeech, dim)
    public int Dim { get; }

    public EmbeddingTables(string refsDir)
    {
        this.Dim = TensorIo.Shape("t3_text_emb_weight")[1];
        this.textEmb = TensorIo.Load(refsDir, "t3_text_emb_weight");
        this.speechEmb = TensorIo.Load(refsDir, "t3_speech_emb_weight");
        this.textPos = TensorIo.Load(refsDir, "t3_text_pos_emb_weight");
        this.speechPos = TensorIo.Load(refsDir, "t3_speech_pos_emb_weight");
    }

    /// <summary>Writes textEmb[token] (+ textPos[pos]) into dest.</summary>
    public void TextEmbedding(int tokenId, int pos, Span<float> dest, bool zeroToken)
    {
        var te = tokenId * this.Dim;
        var pe = pos * this.Dim;
        for (var d = 0; d < this.Dim; d++)
        {
            dest[d] = (zeroToken ? 0f : this.textEmb[te + d]) + this.textPos[pe + d];
        }
    }

    /// <summary>Writes speechEmb[token] + speechPos[pos] into dest.</summary>
    public void SpeechEmbedding(int tokenId, int pos, Span<float> dest)
    {
        var se = tokenId * this.Dim;
        var pe = pos * this.Dim;
        for (var d = 0; d < this.Dim; d++)
        {
            dest[d] = this.speechEmb[se + d] + this.speechPos[pe + d];
        }
    }
}
