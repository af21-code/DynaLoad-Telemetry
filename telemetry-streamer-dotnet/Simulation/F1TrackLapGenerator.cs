using DynaLoad.TelemetryStreamer.Models;

namespace DynaLoad.TelemetryStreamer.Simulation;

/// <summary>
/// Profile B (Motorsport F1 Trackside):
/// Simulates 100 Hz Formula 1 suspension telemetry across a 4-sector track sequence:
/// 1. High-speed straight (quadratic ground-effect aero compression + porpoising harmonic)
/// 2. Heavy braking zone (longitudinal pitch dive & front pushrod load transfer)
/// 3. High-speed chicane (sawtooth kerb strikes exciting High-Speed Bump & Rebound)
/// 4. High-G fast sweeper (sustained lateral roll & vertical aero + mechanical load)
/// </summary>
public sealed class F1TrackLapGenerator : ITelemetryGenerator
{
    public OperationalProfile Profile => OperationalProfile.Formula1Trackside;
    public SimulationScenario Scenario { get; }

    private const double MaxDamperStrokeMm = 65.0;
    private const double EffectiveReservoirLengthMm = 82.0;
    private const double PolytropicGamma = 1.38;

    public F1TrackLapGenerator(SimulationScenario scenario = SimulationScenario.Nominal)
    {
        Scenario = scenario;
    }

    public IReadOnlyList<TelemetryPacket> GenerateSequence(double durationSeconds = 10.0, int sampleRateHz = 100, int seed = 42)
    {
        int totalSamples = Math.Max(10, (int)Math.Round(durationSeconds * sampleRateHz));
        double dt = 1.0 / sampleRateHz;
        var rng = new Random(seed);
        var packets = new List<TelemetryPacket>(totalSamples);

        string scenarioTag = Scenario switch
        {
            SimulationScenario.Nominal => "nominal",
            SimulationScenario.Overload => "overload",
            SimulationScenario.Fault => "fault",
            _ => "nominal"
        };

        double staticGasBar = Scenario switch
        {
            SimulationScenario.Nominal => 16.0,
            SimulationScenario.Overload => 16.0,
            SimulationScenario.Fault => 8.2, // Damper nitrogen reservoir bladder leak
            _ => 16.0
        };

        double prevStrokeMm = 18.0;

        for (int i = 0; i < totalSamples; i++)
        {
            double t = i * dt;
            long timestampMs = (long)Math.Round(t * 1000.0);

            double wheelSpeedKph;
            double pitchDeg;
            double rollDeg;
            double baseVerticalG;
            double baseStrokeMm;

            // Sector progression along a 10s high-speed complex (normalized phase s in [0, 10])
            double s = t % 10.0;

            if (s < 2.5)
            {
                // Sector 1: Main Straight (290 -> 334 kph, quadratic ground-effect aero load)
                double frac = s / 2.5;
                wheelSpeedKph = 290.0 + 44.0 * frac;
                double aeroLoadG = 1.15 * Math.Pow(wheelSpeedKph / 300.0, 2.0);
                double porpoisingOscMm = (Scenario == SimulationScenario.Overload ? 11.5 : 2.8)
                                         * Math.Sin(2.0 * Math.PI * 6.2 * s) * frac;
                double porpoisingG = (Scenario == SimulationScenario.Overload ? 1.45 : 0.22)
                                     * Math.Sin(2.0 * Math.PI * 6.2 * s) * frac;

                baseVerticalG = 1.0 + aeroLoadG + porpoisingG;
                pitchDeg = -0.35 - 0.30 * frac;
                rollDeg = 0.08 * Math.Sin(2.0 * Math.PI * 1.2 * s);
                baseStrokeMm = 16.0 + 24.5 * Math.Pow(wheelSpeedKph / 300.0, 2.0) + porpoisingOscMm;

                if (Scenario == SimulationScenario.Overload && s > 1.7 && s < 2.35)
                {
                    // Severe aero bottoming / titanium skid-block strike at Vmax
                    double bottomPulse = Math.Exp(-Math.Pow((s - 2.05) / 0.08, 2.0));
                    baseStrokeMm = Math.Max(baseStrokeMm, 63.6 * bottomPulse + baseStrokeMm * (1.0 - bottomPulse));
                    baseVerticalG += 2.15 * bottomPulse;
                }
            }
            else if (s < 4.5)
            {
                // Sector 2: Heavy Braking Zone (334 -> 132 kph, longitudinal dive)
                double frac = (s - 2.5) / 2.0;
                wheelSpeedKph = 334.0 - 202.0 * Math.Pow(frac, 0.75);
                pitchDeg = -1.45 * Math.Sin(Math.PI * frac); // Nose dive under 5G longitudinal deceleration
                rollDeg = -0.45 * frac;
                baseVerticalG = 1.65 + 0.55 * (1.0 - frac) + 0.20 * Math.Sin(2.0 * Math.PI * 8.5 * s);
                baseStrokeMm = 38.0 - 11.0 * frac + 2.2 * Math.Sin(2.0 * Math.PI * 8.5 * s);
            }
            else if (s < 7.0)
            {
                // Sector 3: High-Speed Chicane & Kerb Strikes (t = 4.9s apex 1, t = 5.8s apex 2)
                double frac = (s - 4.5) / 2.5;
                wheelSpeedKph = 132.0 + 92.0 * frac;
                pitchDeg = -0.25 + 0.45 * Math.Sin(2.0 * Math.PI * 0.8 * (s - 4.5));
                rollDeg = 1.85 * Math.Sin(2.0 * Math.PI * 0.45 * (s - 4.5));

                // Sawtooth kerb excitations at s = 4.95s and s = 5.85s
                double kerbAmpMm = Scenario switch
                {
                    SimulationScenario.Nominal => 14.5,
                    SimulationScenario.Overload => 28.5, // Aggressive sausage kerb strike causing bottoming
                    SimulationScenario.Fault => 16.5,    // Uncontrolled damper oscillation over kerb
                    _ => 14.5
                };

                double kerbAmpG = Scenario switch
                {
                    SimulationScenario.Nominal => 1.35,
                    SimulationScenario.Overload => 3.45, // Peaks > 5.2 G
                    SimulationScenario.Fault => 1.55,
                    _ => 1.35
                };

                double kerb1 = Math.Exp(-Math.Pow((s - 4.95) / 0.032, 2.0)) * Math.Cos(2.0 * Math.PI * 14.0 * (s - 4.95));
                double kerb2 = Math.Exp(-Math.Pow((s - 5.85) / 0.035, 2.0)) * Math.Cos(2.0 * Math.PI * 13.0 * (s - 5.85));

                baseVerticalG = 1.85 + kerbAmpG * (Math.Abs(kerb1) + 0.85 * Math.Abs(kerb2));
                baseStrokeMm = 33.5 + kerbAmpMm * (kerb1 + 0.9 * kerb2);
            }
            else
            {
                // Sector 4: High-G Fast Sweeper (224 -> 305 kph, sustained cornering load)
                double frac = (s - 7.0) / 3.0;
                wheelSpeedKph = 224.0 + 81.0 * frac;
                pitchDeg = -0.40 - 0.15 * frac;
                rollDeg = 2.05 * Math.Sin(Math.PI * Math.Min(1.0, frac * 1.3));
                baseVerticalG = 2.25 + 0.78 * Math.Sin(Math.PI * frac) + 0.18 * Math.Sin(2.0 * Math.PI * 9.5 * s);
                baseStrokeMm = 36.0 + 11.5 * Math.Sin(Math.PI * frac) + 1.8 * Math.Sin(2.0 * Math.PI * 9.5 * s);
            }

            if (Scenario == SimulationScenario.Fault)
            {
                // Fault Scenario: Hydraulic cavitation & nitrogen loss induce high-frequency 11.5 Hz undamped wheel-hop
                // and deeper baseline sag (+4.5 mm)
                double cavitationHopMm = 4.8 * Math.Sin(2.0 * Math.PI * 11.5 * t);
                baseStrokeMm += 4.5 + cavitationHopMm;
                baseVerticalG += 0.25 * Math.Cos(2.0 * Math.PI * 11.5 * t);
            }

            double strokeMm = Math.Clamp(baseStrokeMm + NextGaussian(rng, 0.18), 1.5, MaxDamperStrokeMm - 0.5);
            double verticalG = baseVerticalG + NextGaussian(rng, 0.03);
            rollDeg += NextGaussian(rng, 0.02);
            pitchDeg += NextGaussian(rng, 0.02);
            wheelSpeedKph += NextGaussian(rng, 0.25);

            double velocityMmPerSec = i == 0 ? 0.0 : (strokeMm - prevStrokeMm) / dt;
            prevStrokeMm = strokeMm;

            // Damper Nitrogen Reservoir Polytropic Law + Shim-Stack Hydraulic Pressure
            double clampedStroke = Math.Min(strokeMm, EffectiveReservoirLengthMm * 0.92);
            double gasPressureBar = staticGasBar * Math.Pow(
                EffectiveReservoirLengthMm / (EffectiveReservoirLengthMm - clampedStroke),
                PolytropicGamma);

            double shimDampingCoeff = Scenario == SimulationScenario.Fault ? 0.0015 : 0.0115;
            double hydraulicDeltaBar = shimDampingCoeff * velocityMmPerSec;
            double internalPressureBar = Math.Max(
                staticGasBar * 0.75,
                gasPressureBar + hydraulicDeltaBar + NextGaussian(rng, 0.15));

            packets.Add(new TelemetryPacket
            {
                TimestampMs = timestampMs,
                VerticalG = Math.Round(verticalG, 4),
                RollDeg = Math.Round(rollDeg, 4),
                PitchDeg = Math.Round(pitchDeg, 4),
                StrutStrokeMm = Math.Round(strokeMm, 4),
                InternalPressureBar = Math.Round(internalPressureBar, 4),
                WheelSpeedKph = Math.Round(wheelSpeedKph, 4),
                Profile = "f1",
                Scenario = scenarioTag
            });
        }

        return packets;
    }

    private static double NextGaussian(Random rng, double stdDev)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return stdDev * randStdNormal;
    }
}
