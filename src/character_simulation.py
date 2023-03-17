import pyray as rl
import pymunk as pm
import numpy as np
import random

from character import Character, character_data_list
from neural_network import NeuralNetwork


class CharacterSimulation:
    def __init__(self, ground_position: tuple[float, float], ground_poly: list[tuple[float, float]]):
        self.collided = False
        self.space: pm.Space = pm.Space()
        self.space.gravity = (0, -900.0)

        self.character: Character = Character(self.space, leg_muscle_strength=1_000_000.0, arm_muscle_strength=50_000.0)

        ground_body: pm.Body = pm.Body(body_type=pm.Body.STATIC)
        ground_body.position = ground_position
        ground_shape = pm.Poly(ground_body, ground_poly)
        ground_shape.friction = 0.8
        ground_shape.collision_type = pm.Body.STATIC

        self.space.add(ground_body, ground_shape)

        self.neural_network: NeuralNetwork = NeuralNetwork()

        self.outputs = np.asarray(character_data_list(self.character))

        self.fitness = 0.0

        self.color = rl.color_from_hsv(random.uniform(0, 360), 0.7, 0.9)

    def collision_detection(self, arbiter, space, data):
        self.collided = True
    def get_Space(self):
        return self.space

    def step(self, time_step: float) -> None:
        self.space.step(time_step)
        inputs = np.asarray(character_data_list(self.character))
        self.outputs = self.neural_network.feedforward(inputs)
        if self.outputs[0] >= 0.5 > self.outputs[1]:
            self.character_move_legs_q()
        if self.outputs[1] >= 0.5 > self.outputs[0]:
            self.character_move_legs_w()
        if self.outputs[2] >= 0.5 > self.outputs[3]:
            self.character_move_knees_o()
        if self.outputs[3] >= 0.5 > self.outputs[2]:
            self.character_move_knees_p()

    def output_data(self) -> dict:
        data = {
            "color": (self.color.r, self.color.g, self.color.b, self.color.a),
            "network": self.neural_network.output_data(),
            "fitness": self.fitness
        }
        return data

    def load_data(self, data: dict) -> None:
        self.color = rl.Color(data["color"][0], data["color"][1], data["color"][2], data["color"][3])
        self.neural_network.load_data(data["network"])

    def character_position(self) -> rl.Vector2:
        return rl.Vector2(self.character.torso.body.position.x, -self.character.torso.body.position.y + 100)

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
