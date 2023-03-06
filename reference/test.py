import pyray, pymunk, sys, os

# Initialize PyRay and Pymunk
pyray.init()
screen = pyray.Screen(600, 600)

space = pymunk.Space()
space.gravity = (0, 500)

# Create two bodies
body_1 = pymunk.Body(10, 100)
body_1.position = (100, 100)

body_2 = pymunk.Body(10, 100)
body_2.position = (300, 100)

# Create two shapes
shape_1 = pymunk.Circle(body_1, 50)
shape_2 = pymunk.Circle(body_2, 50)

# Add the shapes to the space
space.add(body_1, shape_1)
space.add(body_2, shape_2)

# Define a collision handler
def on_collision(arbiter, space, data):
    # Get the shapes that collided
    shape_1, shape_2 = arbiter.shapes

    # Check if the colliding shapes belong to body_1 and body_2
    if (shape_1.body == body_1 and shape_2.body == body_2) or (shape_1.body == body_2 and shape_2.body == body_1):
        # Restart the application
        python = sys.executable
        os.execl(python, python, *sys.argv)

# Add the collision handler to the space
handler = space.add_collision_handler(0, 0)
handler.begin = on_collision

# Run the simulation
while True:
    screen.clear((255, 255, 255))

    # Draw the bodies and shapes
    for body in space.bodies:
        pos = body.position
        angle = body.angle
        pyray.draw_circle(screen, (0, 0, 255), (int(pos.x), int(pos.y)), 50)
        pyray.draw_line(screen, (0, 255, 0), (int(pos.x), int(pos.y)),
                         (int(pos.x + 50 * math.cos(angle)), int(pos.y + 50 * math.sin(angle))), 3)

    # Step the simulation forward
    space.step(1 / 60)

    # Update the PyRay screen
    screen.update()

    # Check for PyRay events
    for event in pyray.get_events():
        if event.type == pyray.QUIT:
            pyray.quit()
            sys.exit()

