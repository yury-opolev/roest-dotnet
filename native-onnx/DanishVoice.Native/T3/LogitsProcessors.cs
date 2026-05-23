namespace DanishVoice.Native.T3;

/// <summary>
/// Faithful ports of the HuggingFace logits processors used in T3.inference, in
/// the same order: repetition penalty -> temperature -> min_p -> top_p.
/// For greedy decoding only the repetition penalty changes the argmax, but all
/// are applied to match the Python pipeline exactly.
/// </summary>
internal static class LogitsProcessors
{
    /// <summary>HF RepetitionPenaltyLogitsProcessor over the already-generated ids.</summary>
    public static void RepetitionPenalty(Span<float> logits, IEnumerable<int> generatedIds, float penalty)
    {
        foreach (var id in generatedIds)
        {
            var s = logits[id];
            logits[id] = s > 0 ? s / penalty : s * penalty;
        }
    }

    public static void Temperature(Span<float> logits, float temperature)
    {
        if (temperature == 1.0f)
        {
            return;
        }
        for (var i = 0; i < logits.Length; i++)
        {
            logits[i] /= temperature;
        }
    }

    /// <summary>
    /// HF MinPLogitsWarper: keep tokens with prob >= min_p * maxProb; filter rest
    /// to -inf. Probabilities are softmax over logits.
    /// </summary>
    public static void MinP(Span<float> logits, float minP)
    {
        var probs = Softmax(logits);
        var maxProb = 0f;
        foreach (var p in probs)
        {
            if (p > maxProb)
            {
                maxProb = p;
            }
        }
        var threshold = minP * maxProb;
        for (var i = 0; i < logits.Length; i++)
        {
            if (probs[i] < threshold)
            {
                logits[i] = float.NegativeInfinity;
            }
        }
    }

    /// <summary>
    /// HF TopPLogitsWarper: keep the smallest set of tokens whose cumulative prob
    /// exceeds top_p (sorted descending); filter the rest. min_tokens_to_keep=1.
    /// </summary>
    public static void TopP(Span<float> logits, float topP)
    {
        var n = logits.Length;
        var probs = Softmax(logits);
        var idx = new int[n];
        for (var i = 0; i < n; i++)
        {
            idx[i] = i;
        }
        Array.Sort(idx, (a, b) => probs[b].CompareTo(probs[a])); // descending

        double cum = 0;
        var keep = new bool[n];
        for (var rank = 0; rank < n; rank++)
        {
            var t = idx[rank];
            keep[t] = true;
            cum += probs[t];
            // HF removes tokens with cumulative prob above top_p, shifting right
            // so the first token over the threshold is kept (min_tokens_to_keep=1).
            if (cum > topP)
            {
                break;
            }
        }
        for (var i = 0; i < n; i++)
        {
            if (!keep[i])
            {
                logits[i] = float.NegativeInfinity;
            }
        }
    }

    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        var best = 0;
        var bestVal = float.NegativeInfinity;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] > bestVal)
            {
                bestVal = logits[i];
                best = i;
            }
        }
        return best;
    }

    private static float[] Softmax(ReadOnlySpan<float> logits)
    {
        var max = float.NegativeInfinity;
        foreach (var l in logits)
        {
            if (l > max)
            {
                max = l;
            }
        }
        var outp = new float[logits.Length];
        double sum = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            var e = Math.Exp(logits[i] - max);
            outp[i] = (float)e;
            sum += e;
        }
        for (var i = 0; i < outp.Length; i++)
        {
            outp[i] = (float)(outp[i] / sum);
        }
        return outp;
    }
}
