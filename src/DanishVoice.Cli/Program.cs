using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace DanishVoice.Cli;

internal static class Program
{
    private const string DefaultServer = "http://localhost:8000";
    private const string DefaultVoice = "mic";
    private const string DefaultOut = "out.wav";

    private static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        var options = ParseArgs(args);
        if (options is null)
        {
            return 1;
        }

        using var http = new HttpClient { BaseAddress = new Uri(options.Server) };
        http.Timeout = TimeSpan.FromMinutes(5);

        var request = new SynthesizeRequest(options.Text, options.Voice);

        Console.WriteLine($"POST {options.Server}/synthesize  voice={options.Voice}  textLen={options.Text.Length}");
        HttpResponseMessage response;
        try
        {
            response = await http.PostAsJsonAsync("/synthesize", request).ConfigureAwait(false);
        }
        catch (HttpRequestException ex)
        {
            Console.Error.WriteLine($"ERROR: could not reach {options.Server} — {ex.Message}");
            Console.Error.WriteLine("Hint: is `docker compose up` running?");
            return 2;
        }

        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            Console.Error.WriteLine($"ERROR: server returned {(int)response.StatusCode} {response.ReasonPhrase}");
            if (!string.IsNullOrWhiteSpace(body))
            {
                Console.Error.WriteLine(body);
            }
            return 3;
        }

        var bytes = await response.Content.ReadAsByteArrayAsync().ConfigureAwait(false);
        await File.WriteAllBytesAsync(options.Out, bytes).ConfigureAwait(false);

        Console.WriteLine($"Wrote {bytes.Length:N0} bytes to {Path.GetFullPath(options.Out)}");
        return 0;
    }

    private static CliOptions? ParseArgs(string[] args)
    {
        string? text = null;
        var voice = DefaultVoice;
        var outPath = DefaultOut;
        var server = DefaultServer;

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            switch (arg)
            {
                case "--voice":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("ERROR: --voice requires a value (mic|nic)");
                        return null;
                    }
                    voice = args[i];
                    break;
                case "--out":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("ERROR: --out requires a value");
                        return null;
                    }
                    outPath = args[i];
                    break;
                case "--server":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("ERROR: --server requires a value");
                        return null;
                    }
                    server = args[i];
                    break;
                default:
                    if (text is not null)
                    {
                        Console.Error.WriteLine($"ERROR: unexpected positional argument: {arg}");
                        return null;
                    }
                    text = arg;
                    break;
            }
        }

        if (string.IsNullOrWhiteSpace(text))
        {
            Console.Error.WriteLine("ERROR: text argument is required");
            PrintUsage();
            return null;
        }

        return new CliOptions(text, voice, outPath, server);
    }

    private static void PrintUsage()
    {
        Console.WriteLine("DanishVoice.Cli — Danish text-to-speech (CoRal Røst-v3 via local server)");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  DanishVoice.Cli <text> [--voice mic|nic] [--out <path>] [--server <url>]");
        Console.WriteLine();
        Console.WriteLine($"Defaults: --voice {DefaultVoice}  --out {DefaultOut}  --server {DefaultServer}");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  DanishVoice.Cli \"Hej, hvordan går det?\"");
        Console.WriteLine("  DanishVoice.Cli \"Hej!\" --voice nic --out greeting.wav");
    }

    private sealed record CliOptions(string Text, string Voice, string Out, string Server);

    private sealed record SynthesizeRequest(
        [property: JsonPropertyName("text")] string Text,
        [property: JsonPropertyName("voice")] string Voice);
}
