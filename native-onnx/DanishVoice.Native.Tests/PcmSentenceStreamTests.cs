namespace DanishVoice.Native.Tests;

public class PcmSentenceStreamTests
{
    // Fake synth: returns N float samples where N = sentence length, so chunk
    // byte length is 2 * sentence.Length — distinguishable per sentence.
    private static float[] FakeSynth(string sentence, CancellationToken ct) =>
        new float[sentence.Length];

    [Fact]
    public async Task StreamAsync_YieldsOneChunkPerSentenceInOrder()
    {
        string[] sentences = ["ab", "cde", "f"];
        var chunks = new List<byte[]>();
        await foreach (var c in PcmSentenceStream.StreamAsync(sentences, FakeSynth))
        {
            chunks.Add(c);
        }

        Assert.Equal(3, chunks.Count);
        Assert.Equal(4, chunks[0].Length);  // "ab" -> 2 samples -> 4 bytes
        Assert.Equal(6, chunks[1].Length);  // "cde" -> 3 samples -> 6 bytes
        Assert.Equal(2, chunks[2].Length);  // "f" -> 1 sample -> 2 bytes
    }

    [Fact]
    public async Task StreamAsync_EmptyList_YieldsNothing()
    {
        var chunks = new List<byte[]>();
        await foreach (var c in PcmSentenceStream.StreamAsync([], FakeSynth))
        {
            chunks.Add(c);
        }

        Assert.Empty(chunks);
    }

    [Fact]
    public async Task StreamAsync_ProducerException_SurfacesToConsumer()
    {
        string[] sentences = ["ok", "boom", "never"];
        float[] Synth(string s, CancellationToken ct) =>
            s == "boom" ? throw new InvalidOperationException("synth failed") : new float[s.Length];

        var chunks = new List<byte[]>();
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var c in PcmSentenceStream.StreamAsync(sentences, Synth))
            {
                chunks.Add(c);
            }
        });

        Assert.Equal("synth failed", ex.Message);
        Assert.Single(chunks); // only the first ("ok") made it through
    }

    [Fact]
    public async Task StreamAsync_PreCancelledToken_DoesNotComplete()
    {
        string[] sentences = ["a", "b"];
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var c in PcmSentenceStream.StreamAsync(sentences, FakeSynth, cancellationToken: cts.Token))
            {
            }
        });
    }
}
