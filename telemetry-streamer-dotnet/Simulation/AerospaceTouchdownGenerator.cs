using DynaLoad.TelemetryStreamer.Models;

namespace DynaLoad.TelemetryStreamer.Simulation;

/// <summary>
/// Profile A (Aerospace MRO - ATA Chapter 32):
/// Simulates 100 Hz Quick Access Recorder (QAR) & Landing Gear Health Monitoring telemetry
/// during approach flare, Weight-on-Wheels (WoW) main gear touchdown, nose-gear derotation, and rollout.
/// </summary>
public sealed class AerospaceTouchdownGenerator : ITelemetryGenerator
{
    public OperationalProfile Profile => OperationalProfile.AerospaceAta32;
    public SimulationScenario Scenario { get; }

    // Main Landing Gear (MLG) Oleo-Pneumatic Strut Physical Constants
    private const double MaxMechanicalStrokeMm = 450.0;
    private const double EffectiveGasColumnLengthMm = 480.0;
    private const double PolytropicGamma = 1.35;

    public AerospaceTouchdownGenerator(SimulationScenario scenario = SimulationScenario.Nominal)
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

        // Scenario-dependent parameters
        double staticPrechargeBar = Scenario switch
        {
            SimulationScenario.Nominal => 30.0,
            SimulationScenario.Overload => 30.0,
            SimulationScenario.Fault => 15.5, // Nitrogen pre-charge leak across upper dynamic seal
            _ => 30.0
        };

        double peakImpactGDelta = Scenario switch
        {
            SimulationScenario.Nominal => 0.48,  // Total peak ~ 1.48 G (well within 1.85 G advisory / 2.10 G EASA Phase 1)
            SimulationScenario.Overload => 1.82, // Total peak ~ 2.82 G (exceeds EASA CS-25 / AMM 05-51-11 Phase 2 2.60 G threshold)
            SimulationScenario.Fault => 0.65,    // Moderate 1.65 G touchdown, but degraded damping & low N2 precharge
            _ => 0.48
        };

        double settledTaxiStrokeMm = Scenario switch
        {
            SimulationScenario.Nominal => 225.0,
            SimulationScenario.Overload => 255.0,
            SimulationScenario.Fault => 295.0, // Excessive static sag due to N2 gas loss
            _ => 225.0
        };

        double dynamicOvershootMm = Scenario switch
        {
            SimulationScenario.Nominal => 68.0,  // Peak stroke ~ 265 mm
            SimulationScenario.Overload => 210.0, // Peak stroke ~ 444 mm (bottoming against mechanical stop)
            SimulationScenario.Fault => 62.0,    // Peak stroke ~ 330 mm with under-damped ringing
            _ => 68.0
        };

        double dampingRatioZeta = Scenario switch
        {
            SimulationScenario.Nominal => 0.62,
            SimulationScenario.Overload => 0.48,
            SimulationScenario.Fault => 0.16, // Severe orifice damping loss / oil foaming
            _ => 0.62
        };

        double touchdownBankDeg = Scenario switch
        {
            SimulationScenario.Overload => 5.9, // High-roll crabbed touchdown (exceeds 5.0 deg asymmetric gear limit)
            SimulationScenario.Fault => 1.8,
            _ => 0.9
        };

        const double touchdownTimeSec = 2.0;
        const double naturalFreqRad = 2.0 * Math.PI * 1.85; // 1.85 Hz main gear heave mode

        double prevStrokeMm = 0.0;

        for (int i = 0; i < totalSamples; i++)
        {
            double t = i * dt;
            long timestampMs = (long)Math.Round(t * 1000.0);

            double verticalG;
            double rollDeg;
            double pitchDeg;
            double strokeMm;
            double wheelSpeedKph;

            if (t < touchdownTimeSec)
            {
                // Phase 1: Final Approach & Flare (Weight-off-Wheels)
                double flareProgress = t / touchdownTimeSec;
                verticalG = 1.00 + 0.04 * Math.Sin(2.0 * Math.PI * 0.7 * t) + NextGaussian(rng, 0.008);
                rollDeg = touchdownBankDeg * (0.6 + 0.4 * flareProgress) + NextGaussian(rng, 0.05);
                pitchDeg = 4.2 + 1.6 * flareProgress + NextGaussian(rng, 0.04);
                strokeMm = Math.Max(0.0, NextGaussian(rng, 0.15));
                wheelSpeedKph = 0.0;
            }
            else
            {
                double tau = t - touchdownTimeSec;

                // Primary touchdown impact pulse + damped structural heave response
                double impactPulse = peakImpactGDelta * Math.Exp(-Math.Pow((tau - 0.18) / 0.11, 2.0));
                double dampedRingG = 0.28 * peakImpactGDelta * Math.Exp(-dampingRatioZeta * naturalFreqRad * tau)
                                     * Math.Cos(naturalFreqRad * Math.Sqrt(1.0 - dampingRatioZeta * dampingRatioZeta) * (tau - 0.18));

                // Secondary nose-gear touchdown bump around t = 3.8s (tau = 1.8s)
                double noseGearPulse = 0.16 * Math.Exp(-Math.Pow((tau - 1.80) / 0.14, 2.0));

                verticalG = 1.00 + impactPulse + dampedRingG + noseGearPulse + NextGaussian(rng, 0.012);

                // Oleo-strut compression kinematics (C1-smooth second-order onset at Weight-on-Wheels)
                double rampUp = 1.0 - (1.0 + tau / 0.055) * Math.Exp(-tau / 0.055);
                double primaryImpactStroke = dynamicOvershootMm * Math.Exp(-Math.Pow((tau - 0.16) / 0.09, 2.0));
                double dampedOscillation = (0.55 * dynamicOvershootMm) * Math.Exp(-dampingRatioZeta * naturalFreqRad * tau)
                                           * Math.Sin(naturalFreqRad * Math.Sqrt(1.0 - dampingRatioZeta * dampingRatioZeta) * tau);
                double rolloutVibration = 2.8 * Math.Sin(2.0 * Math.PI * 4.2 * tau) * Math.Exp(-0.15 * tau);

                strokeMm = settledTaxiStrokeMm * rampUp + primaryImpactStroke + Math.Max(-45.0, dampedOscillation) + rolloutVibration + NextGaussian(rng, 0.25);
                strokeMm = Math.Clamp(strokeMm, 0.0, MaxMechanicalStrokeMm - 3.0);

                // Roll attitude settles from initial bank angle to wings-level on runway
                rollDeg = touchdownBankDeg * Math.Exp(-tau / 0.65) * Math.Cos(3.5 * tau) + NextGaussian(rng, 0.04);

                // Pitch attitude derotates from +5.8 deg flare to +0.2 deg nose-on-runway
                double derotationFactor = 1.0 / (1.0 + Math.Exp((tau - 1.45) / 0.28));
                pitchDeg = 0.2 + 5.6 * derotationFactor + NextGaussian(rng, 0.04);

                // Wheel spin-up at Weight-on-Wheels (0 -> 252 kph within 0.25 s, then autobrake deceleration)
                double spinUp = 1.0 - Math.Exp(-tau / 0.06);
                double autobrakeDecel = Math.Max(0.0, tau - 0.5) * 16.5;
                wheelSpeedKph = Math.Max(60.0, (252.0 - autobrakeDecel) * spinUp + NextGaussian(rng, 0.35));
            }

            // Compute instantaneous stroke velocity (mm/s) for orifice hydraulic pressure coupling
            double strokeVelocityMmPerSec = i == 0 ? 0.0 : (strokeMm - prevStrokeMm) / dt;
            prevStrokeMm = strokeMm;

            // Polytropic Nitrogen Gas Law: P_gas(x) = P_0 * (L_0 / (L_0 - x))^gamma
            double clampedStroke = Math.Min(strokeMm, EffectiveGasColumnLengthMm * 0.94);
            double gasPressureBar = staticPrechargeBar * Math.Pow(
                EffectiveGasColumnLengthMm / (EffectiveGasColumnLengthMm - clampedStroke),
                PolytropicGamma);

            // Hydraulic orifice dynamic pressure drop contribution during rapid compression (relief-valve bounded)
            double orificeCoeff = Scenario == SimulationScenario.Fault ? 0.000004 : 0.000018;
            double hydraulicTransientBar = Math.Clamp(
                orificeCoeff * strokeVelocityMmPerSec * Math.Abs(strokeVelocityMmPerSec),
                -25.0,
                55.0);
            double internalPressureBar = Math.Max(
                staticPrechargeBar * 0.85,
                gasPressureBar + hydraulicTransientBar + NextGaussian(rng, 0.25));

            packets.Add(new TelemetryPacket
            {
                TimestampMs = timestampMs,
                VerticalG = Math.Round(verticalG, 4),
                RollDeg = Math.Round(rollDeg, 4),
                PitchDeg = Math.Round(pitchDeg, 4),
                StrutStrokeMm = Math.Round(strokeMm, 4),
                InternalPressureBar = Math.Round(internalPressureBar, 4),
                WheelSpeedKph = Math.Round(wheelSpeedKph, 4),
                Profile = "aero",
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
