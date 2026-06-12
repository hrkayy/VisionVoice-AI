def risk_level(objects, positions):
    """
    Computes risk level based on detected objects and their proximity:
    - Empty space (no objects): LOW RISK
    - Immediate proximity danger (any object occupying > 50% of screen height): HIGH RISK
    - Person detected: HIGH RISK
    - Table/Other Obstacles detected: MEDIUM RISK
    """
    if not objects or not positions:
        return "LOW RISK"  # Empty space
        
    # Proximity check: if any object is extremely close to the user, escalate to HIGH RISK
    close_obstacles = [p for p in positions if p["y_height"] > 0.50]
    if close_obstacles:
        return "HIGH RISK"
        
    if "person" in objects:
        return "HIGH RISK"
        
    if "table" in objects:
        return "MEDIUM RISK"
        
    return "MEDIUM RISK"
