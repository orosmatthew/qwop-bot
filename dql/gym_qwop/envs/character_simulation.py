import pyray as rl
import pymunk as pm
import numpy as np
import random

from dql.gym_qwop.envs.character import Character, character_data_list



class CharacterSimulation:
    def __init__(self):
        self.collided = False
        self.space: pm.Space = pm.Space()
        self.space.gravity = (0, -900.0)

        self.ground_position = 50, 150
        self.ground_poly = [
            (-50000, -25),
            (-50000, 25),
            (50000, 25),
            (50000, -25),
        ]

        self.character: Character = Character(self.space, leg_muscle_strength=1_000_000.0, arm_muscle_strength=50_000.0)

        ground_body: pm.Body = pm.Body(body_type=pm.Body.STATIC)
        ground_body.position = self.ground_position
        ground_shape = pm.Poly(ground_body, self.ground_poly)
        ground_shape.friction = 0.8
        ground_shape.collision_type = pm.Body.STATIC

        self.space.add(ground_body, ground_shape)

        self.outputs = np.asarray(character_data_list(self.character))

        self.fitness = 0.0

        self.color = rl.color_from_hsv(random.uniform(0, 360), 0.7, 0.9)

        self.handler = self.space.add_collision_handler(1, 2)

    def collision_detection(self, arbiter, space, data):
        self.collided = True

    def step(self, time_step: float) -> None:
        self.space.step(time_step)
        inputs = np.asarray(character_data_list(self.character))
        # self.outputs = self.neural_network.feedforward(inputs)
        # if self.outputs[0] >= 0.5 > self.outputs[1]:
        #     self.character_move_legs_q()
        # if self.outputs[1] >= 0.5 > self.outputs[0]:
        #     self.character_move_legs_w()
        # if self.outputs[2] >= 0.5 > self.outputs[3]:
        #     self.character_move_knees_o()
        # if self.outputs[3] >= 0.5 > self.outputs[2]:
        #     self.character_move_knees_p()

    # def output_data(self) -> dict:
    #     data = {
    #         "color": (self.color.r, self.color.g, self.color.b, self.color.a),
    #         "network": self.neural_network.output_data(),
    #         "fitness": self.fitness
    #     }
    #     return data

    # def load_data(self, data: dict) -> None:
    #     self.color = rl.Color(data["color"][0], data["color"][1], data["color"][2], data["color"][3])
    #     self.neural_network.load_data(data["network"])

    def render(self):
        rl.set_target_fps(60)
        camera = rl.Camera2D(rl.Vector2(1280 / 2, 720 / 2), rl.Vector2(0, 0), 0.0, 1.0)
        rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)
        rl.init_window(1280, 720, "QWOP-BOT")

        ground_body: pm.Body = pm.Body(body_type=pm.Body.STATIC)
        ground_body.position = self.ground_position
        ground_shape = pm.Poly(ground_body, self.ground_poly)
        ground_shape.friction = 0.8
        ground_shape.collision_type = pm.Body.STATIC
        ground_shape.collision_type = 2

        sim_time: float = 0.0
        app_time: float = 0.0
        sub_sim_time: float = 0.0
        time_step = 1.0 / 60.0

        sim = CharacterSimulation()

        while not rl.window_should_close():
            rl.begin_drawing()
            rl.begin_mode_2d(camera)

            sim.step(time_step)
            sim_time += time_step
            app_time += time_step

            rl.clear_background(rl.BLACK)
            sim.draw_character()
            camera.target = sim.character_position()

            rl.draw_rectangle_pro(
                rl.Rectangle(round(ground_body.position.x), round(-ground_body.position.y), 50000, 50),
                rl.Vector2(50000 / 2, 50 / 2), 0.0, rl.GREEN)

            if rl.is_key_down(rl.KeyboardKey.KEY_Q):
                sim.character_move_legs_q()
            elif rl.is_key_down(rl.KeyboardKey.KEY_W):
                sim.character_move_legs_w()
            if rl.is_key_down(rl.KeyboardKey.KEY_O):
                sim.character_move_knees_o()
            elif rl.is_key_down(rl.KeyboardKey.KEY_P):
                sim.character_move_knees_p()

            rl.end_mode_2d()

            max_x = -float('inf')

            sim.handler.separate = sim.collision_detection
            if sim.character_position().x > max_x:
                max_x = sim.character_position().x

            rl.draw_text("Max Distance: " + str(round(max_x, 0) / 1000.0) + "m", 20, 0, 50,
                         rl.Color(153, 204, 255, 255))

            rl.end_drawing()
        rl.close_window()
    def character_position(self) -> rl.Vector2:
        return rl.Vector2(self.character.torso.body.position.x, -self.character.torso.body.position.y + 100)

    def action(self, action):
        if action == 0:
            self.character.move_legs_q()
        if action == 1:
            self.character.move_legs_w()
        if action == 2:
            self.character.move_knees_o()
        if action == 3:
            self.character.move_knees_p()

    def draw_character(self) -> None:
        self.character.draw(self.color, self.collided)

    def character_move_legs_q(self) -> None:
        self.character.move_legs_q()

    def character_move_legs_w(self) -> None:
        self.character.move_legs_w()

    def character_move_knees_o(self) -> None:
        self.character.move_knees_o()

    def character_move_knees_p(self) -> None:
        self.character.move_knees_p()

