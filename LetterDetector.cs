using System.Collections.Generic;
using Tracking4All;
using Unity.Sentis;
using UnityEngine;

public static class LetterDetector
{
    // MediaPipe Pose landmark indices (only the ones used for angle computation)
    private const int LeftShoulder = 11;
    private const int RightShoulder = 12;
    private const int LeftElbow = 13;
    private const int RightElbow = 14;
    private const int LeftWrist = 15;
    private const int RightWrist = 16;
    private const int LeftHip = 23;
    private const int RightHip = 24;
    private const int LeftKnee = 25;
    private const int RightKnee = 26;
    private const int LeftAnkle = 27;
    private const int RightAnkle = 28;

    // 14 angle definitions matching the Python training order exactly.
    // Each tuple is (point1, vertex, point2) — the angle is measured at the vertex.
    private static readonly (int p1, int vertex, int p2)[] AngleDefinitions =
    {
        (LeftShoulder,  LeftElbow,     LeftWrist),      // left_elbow
        (RightShoulder, RightElbow,    RightWrist),     // right_elbow
        (LeftHip,       LeftShoulder,  LeftElbow),      // left_shoulder
        (RightHip,      RightShoulder, RightElbow),     // right_shoulder
        (LeftShoulder,  LeftHip,       LeftKnee),       // left_hip
        (RightShoulder, RightHip,      RightKnee),      // right_hip
        (LeftHip,       LeftKnee,      LeftAnkle),      // left_knee
        (RightHip,      RightKnee,     RightAnkle),     // right_knee
        (RightShoulder, LeftShoulder,  LeftElbow),      // left_armpit
        (LeftShoulder,  RightShoulder, RightElbow),     // right_armpit
        (LeftAnkle,     LeftHip,       LeftShoulder),   // left_side_bend
        (RightAnkle,    RightHip,      RightShoulder),  // right_side_bend
        (LeftAnkle,     LeftKnee,      LeftHip),        // left_leg_chain
        (RightAnkle,    RightKnee,     RightHip),       // right_leg_chain
    };

    private static Vector2 ToXY(Landmark landmark)
    {
        return new Vector2(landmark.Position.x, landmark.Position.y);
    }

    private static float ComputeAngleDegrees(Vector2 p1, Vector2 vertex, Vector2 p2)
    {
        Vector2 v1 = p1 - vertex;
        Vector2 v2 = p2 - vertex;
        float dot = Vector2.Dot(v1, v2);
        float cross = v1.x * v2.y - v1.y * v2.x;
        return Mathf.Atan2(cross, dot) * Mathf.Rad2Deg;
    }

    /// <summary>
    /// Computes the 14 joint angles from pose landmarks.
    /// Uses only the x and y components to match the 2D training data.
    /// </summary>
    public static float[] ComputeAngles(Landmark[] landmarks)
    {
        float[] angles = new float[AngleDefinitions.Length];
        for (int i = 0; i < AngleDefinitions.Length; i++)
        {
            var (p1, vertex, p2) = AngleDefinitions[i];
            angles[i] = ComputeAngleDegrees(ToXY(landmarks[p1]), ToXY(landmarks[vertex]), ToXY(landmarks[p2]));
        }
        return angles;
    }

    /// <summary>
    /// Runs each ONNX model on the given pose landmarks and returns
    /// the confidence (positive-class probability) for each model.
    /// </summary>
    /// <param name="landmarks">
    /// The 33-element Landmark array from GetAllPoseLandmarks.GetAllLandmarks().
    /// </param>
    /// <param name="models">
    /// ONNX letter models loaded as IWorker instances (one per letter).
    /// </param>
    /// <returns>Confidence values in the same order as the input models list.</returns>
    public static float[] Detect(Landmark[] landmarks, IReadOnlyList<IWorker> models)
    {
        float[] angles = ComputeAngles(landmarks);
        float[] confidences = new float[models.Count];

        using var input = new Tensor<float>(new TensorShape(1, angles.Length), angles);

        for (int i = 0; i < models.Count; i++)
        {
            models[i].Schedule(input);
            var output = models[i].PeekOutput("output_probability") as Tensor<float>;
            using var cpuOutput = output.ReadbackAndClone();
            confidences[i] = cpuOutput[0, 1]; // index 1 = positive class
        }

        return confidences;
    }
}
