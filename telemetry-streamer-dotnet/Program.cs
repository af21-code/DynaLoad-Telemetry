using System.Globalization;
using DynaLoad.TelemetryStreamer.Models;
using DynaLoad.TelemetryStreamer.Networking;
using DynaLoad.TelemetryStreamer.Simulation;

namespace DynaLoad.TelemetryStreamer;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

        string profileArg = "aero";
        string scenarioArg = "nominal";
        double durationSec = 10.0;
        int sampleRateHz = 100;
        string host = "127.0.0.1";
        int port = 5005;
        bool fastMode = false;
        bool selfTest = false;
        string? exportCsvPath = null;
        string? exportJsonPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--profile" or "-p" when i + 1 < args.Length:
                    profileArg = args[++i];
                    break;
                case "--scenario" or "-s" when i + 1 < args.Length:
                    scenarioArg = args[++i];
                    break;
                case "--duration-sec" or "-d" when i + 1 < args.Length:
                    durationSec = double.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--rate-hz" or "-r" when i + 1 < args.Length:
                    sampleRateHz = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--host" when i + 1 < args.Length:
                    host = args[++i];
                    break;
                case "--port" when i + 1 < args.Length:
                    port = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--export-csv" when i + 1 < args.Length:
                    exportCsvPath = args[++i];
                    break;
                case "--export-json" when i + 1 < args.Length:
                    exportJsonPath = args[++i];
                    break;
                case "--fast":
                    fastMode = true;
                    break;
                case "--self-test":
                    selfTest = true;
                    break;
                case "--help" or "-h":
                    PrintUsage();
                    return 0;
            }
        }

        if (selfTest)
        {
            return RunSelfTest(sampleRateHz);
        }

        OperationalProfile profile = profileArg.ToLowerInvariant() switch
        {
            "aero" or "aerospace" or "ata32" => OperationalProfile.AerospaceAta32,
            "f1" or "motorsport" or "formula1" => OperationalProfile.Formula1Trackside,
            _ => throw new ArgumentException($"Unsupported profile '{profileArg}'. Expected 'aero' or 'f1'.")
        };

        SimulationScenario scenario = scenarioArg.ToLowerInvariant() switch
        {
            "nominal" => SimulationScenario.Nominal,
            "overload" or "hardlanding" or "kerb" => SimulationScenario.Overload,
            "fault" or "degraded" or "leak" => SimulationScenario.Fault,
            _ => throw new ArgumentException($"Unsupported scenario '{scenarioArg}'. Expected 'nominal', 'overload', or 'fault'.")
        };

        ITelemetryGenerator generator = profile switch
        {
            OperationalProfile.AerospaceAta32 => new AerospaceTouchdownGenerator(scenario),
            OperationalProfile.Formula1Trackside => new F1TrackLapGenerator(scenario),
            _ => new AerospaceTouchdownGenerator(scenario)
        };

        IReadOnlyList<TelemetryPacket> packets = generator.GenerateSequence(durationSec, sampleRateHz);

        Console.WriteLine("================================================================================");
        Console.WriteLine("  DYNALOAD-TELEMETRY | 100 Hz Structural Load & Damper Kinematics UDP Streamer");
        Console.WriteLine("================================================================================");
        Console.WriteLine($"  Operational Profile : {(profile == OperationalProfile.AerospaceAta32 ? "Profile A - Aerospace MRO (ATA Chapter 32 Landing Gear)" : "Profile B - Motorsport F1 Trackside (Suspension & Kerb Loads)")}");
        Console.WriteLine($"  Simulation Scenario : {scenario}");
        Console.WriteLine($"  Sampling Frequency  : {sampleRateHz} Hz (dt = {1000.0 / sampleRateHz:F1} ms)");
        Console.WriteLine($"  Total Frames        : {packets.Count} ({durationSec:F1} s)");
        Console.WriteLine($"  UDP Target Endpoint : udp://{host}:{port} (Real-time pacing: {!fastMode})");
        Console.WriteLine("--------------------------------------------------------------------------------");

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };

        using var publisher = new UdpTelemetryPublisher(host, port);
        int stepReport = Math.Max(1, packets.Count / 5);

        int transmitted = await publisher.StreamPacketsAsync(
            packets,
            sampleRateHz: sampleRateHz,
            realTimePacing: !fastMode,
            onProgress: (idx, pkt) =>
            {
                if (idx == 0 || (idx + 1) % stepReport == 0 || idx == packets.Count - 1)
                {
                    Console.WriteLine(
                        $"  [Frame {idx + 1,4}/{packets.Count}] t={pkt.TimestampMs,5} ms | " +
                        $"n_z={pkt.VerticalG,6:F2} G | stroke={pkt.StrutStrokeMm,6:F1} mm | " +
                        $"P_int={pkt.InternalPressureBar,6:F1} bar | V_wheel={pkt.WheelSpeedKph,5:F1} kph");
                }
            },
            cancellationToken: cts.Token);

        if (!string.IsNullOrWhiteSpace(exportCsvPath))
        {
            UdpTelemetryPublisher.ExportToCsv(packets, exportCsvPath);
            Console.WriteLine($"  [+] Exported CSV telemetry to: {Path.GetFullPath(exportCsvPath)}");
        }

        if (!string.IsNullOrWhiteSpace(exportJsonPath))
        {
            UdpTelemetryPublisher.ExportToJson(packets, exportJsonPath);
            Console.WriteLine($"  [+] Exported JSON telemetry to: {Path.GetFullPath(exportJsonPath)}");
        }

        double maxG = packets.Max(p => p.VerticalG);
        double maxStroke = packets.Max(p => p.StrutStrokeMm);
        double maxPressure = packets.Max(p => p.InternalPressureBar);

        Console.WriteLine("--------------------------------------------------------------------------------");
        Console.WriteLine($"  Stream Complete: {transmitted} UDP datagrams transmitted.");
        Console.WriteLine($"  Envelope Summary: Peak Vertical G = {maxG:F2} G | Max Stroke = {maxStroke:F1} mm | Peak Pressure = {maxPressure:F1} bar");
        Console.WriteLine("================================================================================");

        return 0;
    }

    private static int RunSelfTest(int sampleRateHz)
    {
        Console.WriteLine("[Self-Test] Validating all 6 Profile x Scenario 100 Hz generators...");
        var profiles = new[] { OperationalProfile.AerospaceAta32, OperationalProfile.Formula1Trackside };
        var scenarios = new[] { SimulationScenario.Nominal, SimulationScenario.Overload, SimulationScenario.Fault };

        foreach (OperationalProfile profile in profiles)
        {
            foreach (SimulationScenario scenario in scenarios)
            {
                ITelemetryGenerator gen = profile == OperationalProfile.AerospaceAta32
                    ? new AerospaceTouchdownGenerator(scenario)
                    : new F1TrackLapGenerator(scenario);

                IReadOnlyList<TelemetryPacket> packets = gen.GenerateSequence(durationSeconds: 5.0, sampleRateHz: sampleRateHz);
                if (packets.Count != 5 * sampleRateHz)
                {
                    Console.Error.WriteLine($"[FAIL] Expected {5 * sampleRateHz} packets, got {packets.Count}");
                    return 1;
                }

                // Verify 10 ms step at 100 Hz
                for (int i = 1; i < packets.Count; i++)
                {
                    long dtMs = packets[i].TimestampMs - packets[i - 1].TimestampMs;
                    if (dtMs != 1000 / sampleRateHz)
                    {
                        Console.Error.WriteLine($"[FAIL] Non-monotonic or invalid dt_ms={dtMs} at index {i}");
                        return 1;
                    }
                }

                double peakG = packets.Max(p => p.VerticalG);
                double peakStroke = packets.Max(p => p.StrutStrokeMm);
                double minPressure = packets.Min(p => p.InternalPressureBar);
                string sampleJson = packets[0].ToJson();

                if (!sampleJson.Contains("\"timestamp_ms\"") ||
                    !sampleJson.Contains("\"vertical_g\"") ||
                    !sampleJson.Contains("\"strut_stroke_mm\"") ||
                    !sampleJson.Contains("\"internal_pressure_bar\"") ||
                    !sampleJson.Contains("\"wheel_speed_kph\""))
                {
                    Console.Error.WriteLine("[FAIL] Telemetry JSON schema missing required properties.");
                    return 1;
                }

                Console.WriteLine(
                    $"  [PASS] {profile,-18} | {scenario,-8} -> {packets.Count} frames @ {sampleRateHz} Hz | " +
                    $"Peak G: {peakG,5:F2} G | Max Stroke: {peakStroke,6:F1} mm | Min P: {minPressure,5:F1} bar");
            }
        }

        Console.WriteLine("[Self-Test] All 6 telemetry streams verified successfully.");
        return 0;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Usage: dotnet run --project telemetry-streamer-dotnet -- [options]");
        Console.WriteLine("Options:");
        Console.WriteLine("  --profile, -p <aero|f1>                Operational profile (default: aero)");
        Console.WriteLine("  --scenario, -s <nominal|overload|fault> Simulation scenario (default: nominal)");
        Console.WriteLine("  --duration-sec, -d <seconds>           Stream duration in seconds (default: 10.0)");
        Console.WriteLine("  --rate-hz, -r <hz>                     Sampling rate in Hz (default: 100)");
        Console.WriteLine("  --host <ip>                            UDP destination host (default: 127.0.0.1)");
        Console.WriteLine("  --port <port>                          UDP destination port (default: 5005)");
        Console.WriteLine("  --export-csv <filepath>                Export generated frames to CSV");
        Console.WriteLine("  --export-json <filepath>               Export generated frames to JSON");
        Console.WriteLine("  --fast                                 Burst mode without 10 ms real-time wait");
        Console.WriteLine("  --self-test                            Run deterministic verification across all modes");
    }
}
