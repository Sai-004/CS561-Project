 # # ✅ Create a small sphere at the bin position (for debugging)
    # marker_radius = 0.5  # Small sphere size
    # marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=marker_radius, rgbaColor=[1, 0, 0, 1])  # Red sphere
    # marker_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=marker_radius)  # Collision shape

    # marker_id = p.createMultiBody(baseMass=0, 
    #                               baseCollisionShapeIndex=marker_collision, 
    #                               baseVisualShapeIndex=marker_visual, 
    #                               basePosition=[pos[0]+3, pos[1]-0.1, 1])  # Slightly above ground