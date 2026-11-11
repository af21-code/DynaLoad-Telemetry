using DynaLoad.TelemetryStreamer.Models;

namespace DynaLoad.TelemetryStreamer.Simulation;

public enum OperationalProfile
{
    AerospaceAta32,
    Formula1Trackside
}

public enum SimulationScenario
{
    Nominal,
    Overload,
    Fault
}

public interface ITelemetryGenerator
{
    OperationalProfile Profile { get; }
    SimulationScenario Scenario { get; }

    /// <summary>
    /// Generates a deterministic sequence of 100 Hz telemetry frames for the specified duration.
    /// </summary>
    IReadOnlyList<TelemetryPacket> GenerateSequence(double durationSeconds = 10.0, int sampleRateHz = 100, int seed = 42);
}
