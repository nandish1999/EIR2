using UnityEngine;

/// <summary>
/// Represents a single butterfly image attached to a leaf cluster node.
/// </summary>
[System.Serializable]
public class ImageItem
{
    // -----------------------------------------------------------
    // Identity
    // -----------------------------------------------------------

    /// <summary>
    /// The butterfly image filename.
    /// </summary>
    public string ImageFileName;

    /// <summary>
    /// Parent node ID.
    /// </summary>
    public string ParentNodeId;

    /// <summary>
    /// Planet index.
    /// </summary>
    public int PlanetIndex;

    // -----------------------------------------------------------
    // Spatial data
    // -----------------------------------------------------------

    /// <summary>
    /// UMAP position.
    /// </summary>
    public Vector3 Position;

    // -----------------------------------------------------------
    // Representative image color
    // -----------------------------------------------------------

    /// <summary>
    /// RGB color parsed from CSV.
    /// Used to color the cluster/planet.
    /// </summary>
    public Color ImageColor;

    // -----------------------------------------------------------
    // Runtime reference
    // -----------------------------------------------------------

    /// <summary>
    /// Parent cluster node.
    /// </summary>
    public ClusterNode ParentNode;

    // -----------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------

    public Vector3 GetLocalPosition()
    {
        if (ParentNode != null)
            return Position - ParentNode.Position;

        return Position;
    }

    public override string ToString()
    {
        return $"[Image] {ImageFileName} parent={ParentNodeId} " +
               $"planet={PlanetIndex} pos=({Position.x:F2},{Position.y:F2},{Position.z:F2})";
    }
}