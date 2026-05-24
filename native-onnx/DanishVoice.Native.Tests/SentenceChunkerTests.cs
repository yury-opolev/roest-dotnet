namespace DanishVoice.Native.Tests;

public class SentenceChunkerTests
{
    [Fact]
    public void Split_EmptyInput_ReturnsEmpty()
    {
        Assert.Empty(SentenceChunker.Split(""));
        Assert.Empty(SentenceChunker.Split("   "));
    }

    [Fact]
    public void Split_ThreeSentences_ReturnsThree()
    {
        var chunks = SentenceChunker.Split("Hej. Hvordan går det? Godt!");
        Assert.Equal(3, chunks.Count);
        Assert.Equal("Hej.", chunks[0]);
        Assert.Equal("Hvordan går det?", chunks[1]);
        Assert.Equal("Godt!", chunks[2]);
    }

    [Fact]
    public void Split_DanishAbbreviation_DoesNotBreak()
    {
        var chunks = SentenceChunker.Split("Vi har bl.a. æbler. Færdig.");
        Assert.Equal(2, chunks.Count);
        Assert.Equal("Vi har bl.a. æbler.", chunks[0]);
        Assert.Equal("Færdig.", chunks[1]);
    }

    [Fact]
    public void Split_SingleInitials_DoNotBreak()
    {
        var chunks = SentenceChunker.Split("H. C. Andersen skrev eventyr.");
        Assert.Single(chunks);
    }

    [Fact]
    public void Split_AddsTerminalPunctuationWhenMissing()
    {
        var chunks = SentenceChunker.Split("Ingen punktum her");
        Assert.Single(chunks);
        Assert.Equal("Ingen punktum her.", chunks[0]);
    }

    [Fact]
    public void Split_OverlongSentence_SplitsAtClauseBoundary()
    {
        var clause = new string('a', 400);
        var text = clause + ", " + clause + ".";
        var chunks = SentenceChunker.Split(text, maxChunkChars: 600);
        Assert.True(chunks.Count >= 2);
        Assert.All(chunks, c => Assert.True(c.Length <= 600));
    }
}
