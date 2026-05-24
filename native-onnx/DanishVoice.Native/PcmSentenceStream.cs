using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace DanishVoice.Native;

/// <summary>
/// Drives per-sentence synthesis with look-ahead: a background producer
/// synthesizes sentences sequentially and pushes PCM chunks through a bounded
/// channel, so the next sentence synthesizes while the caller consumes the
/// current one. Mirrors the consumer project's KokoroTextToSpeech streaming.
/// </summary>
internal static class PcmSentenceStream
{
    public static async IAsyncEnumerable<byte[]> StreamAsync(
        IReadOnlyList<string> sentences,
        Func<string, CancellationToken, float[]> synthSentence,
        int channelCapacity = 3,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var channel = Channel.CreateBounded<byte[]>(new BoundedChannelOptions(channelCapacity)
        {
            SingleReader = true,
            SingleWriter = true,
            FullMode = BoundedChannelFullMode.Wait,
        });

        var producer = Task.Run(async () =>
        {
            try
            {
                foreach (var sentence in sentences)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var samples = synthSentence(sentence, cancellationToken);
                    var pcm = AudioPcm.FloatToPcm16(samples);
                    if (pcm.Length > 0)
                    {
                        await channel.Writer.WriteAsync(pcm, cancellationToken).ConfigureAwait(false);
                    }
                }

                channel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
        }, cancellationToken);

        try
        {
            await foreach (var chunk in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return chunk;
            }
        }
        finally
        {
            channel.Writer.TryComplete();
            // Observe producer faults / cancellation; swallow to avoid masking an
            // exception already surfaced to the consumer via channel completion.
            try
            {
                await producer.ConfigureAwait(false);
            }
            catch
            {
                // Already surfaced through ReadAllAsync (or is the cancellation
                // the consumer is about to observe).
            }
        }
    }
}
