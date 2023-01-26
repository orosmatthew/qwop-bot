from pyray import *
import pymunk.util
from pymunk import Vec2d
from math import degrees

space = pymunk.Space()
space.gravity = (0, -900.0)

pointer_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
pointer_body.position = 100, 300

mass = 10
radius = 25
inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
body = pymunk.Body(mass, inertia)
body.position = 50, 400
shape = pymunk.Circle(body, radius, Vec2d(0, 0))
shape.friction = 0.2
spring = pymunk.constraints.DampedRotarySpring(pointer_body, body, 0, 125000.0, 6000.0)
space.add(body, shape, spring)

ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
ground_body.position = 10, 10
ground_shape = pymunk.Segment(ground_body, (5, 50), (500, 0), 10)
ground_shape.friction = 0.2
space.add(ground_body, ground_shape)

set_target_fps(60)

init_window(1280, 720, "QWOP-BOT")
while not window_should_close():

    pointer_body.position = body.position.x, body.position.y + 100
    pointer_body.angle = (pointer_body.position - body.position).angle
    space.step(1.0 / 60.0)

    begin_drawing()

    clear_background(BLACK)

    draw_circle(int(pointer_body.position.x), int(-pointer_body.position.y + 450), 5, BLUE)
    # draw_circle(int(body.position.x), int(-body.position.y + 450), 25, RED)
    draw_rectangle_pro(Rectangle(int(body.position.x), int(-body.position.y + 450), 25, 25), Vector2(25 / 2, 25 / 2),
                       degrees(body.angle), RED)
    draw_line_ex(Vector2(5, 400), Vector2(500, 450), 40, GREEN)

    if is_key_pressed(KeyboardKey.KEY_R):
        body.position = 50, 400
        body.velocity = 0, 0

    if is_key_down(KeyboardKey.KEY_F):
        body.apply_force_at_world_point((-100000, 10000), body.position)

    if is_key_down(KeyboardKey.KEY_P):
        body.apply_force_at_local_point((-10000, 0), (0, -25))

    end_drawing()
close_window()
