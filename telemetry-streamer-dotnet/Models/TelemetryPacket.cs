using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DynaLoad.TelemetryStreamer.Models;

/// <summary>
/// Represents a single 100 Hz (10 ms interval) structural load and damper kinematics telemetry frame.
/// Shared across Profile A (Aerospace MRO - ATA Chapter 32) and Profile B (Motorsport F1 Trackside).
/// </summary>
public sealed record TelemetryPacket
{
    [JsonPropertyName("timestamp_ms")]
    public required long TimestampMs { get; init; }

    [JsonPropertyName("vertical_g")]
    public required double VerticalG { get; init; }

    [JsonPropertyName("roll_deg")]
    public required double RollDeg { get; init; }

    [JsonPropertyName("pitch_deg")]
    public required double PitchDeg { get; init; }

    [JsonPropertyName("strut_stroke_mm")]
    public required double StrutStrokeMm { get; init; }

    [JsonPropertyName("internal_pressure_bar")]
    public required double InternalPressureBar { get; init; }

    [JsonPropertyName("wheel_speed_kph")]
    public required double WheelSpeedKph { get; init; }

    [JsonPropertyName("profile")]
    public string Profile { get; init; } = "aero";

    [JsonPropertyName("scenario")]
    public string Scenario { get; init; } = "nominal";

    public static string CsvHeader =>
        "timestamp_ms,vertical_g,roll_deg,pitch_deg,strut_stroke_mm,internal_pressure_bar,wheel_speed_kph,profile,scenario";

    public string ToCsvRow() =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"{TimestampMs},{VerticalG:F4},{RollDeg:F4},{PitchDeg:F4},{StrutStrokeMm:F4},{InternalPressureBar:F4},{WheelSpeedKph:F4},{Profile},{Scenario}");

    public string ToJson() => JsonSerializer.Serialize(this);
}
