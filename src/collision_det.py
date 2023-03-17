from main import *
def on_collision(arbiter, space, character):
            # Get the shapes that collided
            shape_1, shape_2 = arbiter.shapes
            list_of_shapes = [character.head.shape,
                              character.torso.shape,
                              character.right_biceps.limb.shape,
                              character.right_forearm.shape,
                              character.left_biceps.limb.shape,
                              character.left_forearm.shape]

            # Check if the colliding shapes belong to the head and floor
            if (shape_1 in list_of_shapes and shape_2 == ground_shape) or (
                    shape_1 == ground_shape and shape_2 in list_of_shapes):
                print("touched")
                return True
            return False