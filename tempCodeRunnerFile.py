_pos, y_pos]) > safe_zone_radius:
                break  # Ensure cylinders do not spawn in the robot's initial area

        cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=0.3)
        cylinder_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cylinder_shape, basePosition=[x_pos, y_pos, 0.15])
        p.changeVisualShape(cylinder_body, -1, rgbaColor=color)
        
        # Assign tokens: (cylinder ID, color token, cylinder number)
        cylinders.append((cylinder_body, color_token, i))