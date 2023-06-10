from multiprocessing import Pool
from typing import Callable, Tuple, Union

from augmax.geometric import GeometricTransformation, LazyCoordinates
import numpy as np
import jax.numpy as jnp
import jax.random as jr


class DataAugmentationFactory:
    """This is a base library to process/transform the elements of a numpy
    array according to a given function.
    """
    def __init__(
        self, 
        processor: Callable,
    ) -> None:
        self.processor = processor

    def __call__(
        self,
        img: np.ndarray,
        configs: Union[dict, list],
        n_processes: int=90,
    ) -> np.ndarray:
        img_processed = \
            self.process_multiple_multiprocessing(img, configs, n_processes)
        
        return img_processed

    def process_single(
        self,
        img: np.ndarray,
        *args: list, 
        **kwargs: dict,
    ) -> np.ndarray:
        img_processed = self.processor(img, *args, **kwargs)
        
        return img_processed

    def process_multiple(
        self,
        imgs: np.ndarray,
        configs: Union[dict, list],
    ) -> np.ndarray:
        imgs_processed = []
        for X, config in zip(imgs, configs):
            X_processed = self.process_single(X, **config)
            imgs_processed.append(X_processed)
        imgs_processed = np.stack(imgs_processed, axis=0)
        
        return imgs_processed

    def process_multiple_multiprocessing(
        self,
        imgs: np.ndarray,
        configs: Union[dict, list],
        n_processes: int,
    ) -> np.ndarray:
        num_elements = len(imgs)
        if isinstance(configs, dict):
            configs = [configs] * num_elements

        if n_processes == 1:
            imgs_processed = self.process_multiple(imgs, configs)
            imgs_processed = imgs_processed.reshape(num_elements, -1)
            
            return imgs_processed

        imgs_processed = np.array_split(imgs, n_processes)
        config_split = np.array_split(configs, n_processes)
        elements = zip(imgs_processed, config_split)

        with Pool(processes=n_processes) as pool:
            imgs_processed = pool.starmap(self.process_multiple, elements)
            imgs_processed = np.concatenate(imgs_processed, axis=0)
        pool.join()
        imgs_processed = imgs_processed.reshape(num_elements, -1)
        
        return imgs_processed
    
    
class Rotate(GeometricTransformation):
    """Rotates the image by a random arbitrary angle.
    Adapted from https://github.com/khdlr/augmax/.
    """
    def __init__(
        self,
        angle_range: Union[Tuple[float, float], float]=(-30, 30),
        prob: float = 1.0
    ):
        super().__init__()
        if not hasattr(angle_range, '__iter__'):
            angle_range = (-angle_range, angle_range)
        self.theta_min, self.theta_max = map(jnp.radians, angle_range)
        self.probability = prob

    def transform_coordinates(
        self, 
        rng: jnp.ndarray, 
        coordinates: LazyCoordinates, 
        invert=False
    ):
        do_apply = jr.bernoulli(rng, self.probability)
        theta = do_apply * jr.uniform(rng, minval=self.theta_min, maxval=self.theta_max)

        if invert:
            theta = -theta

        transform = jnp.array([
            [ jnp.cos(theta), jnp.sin(theta), 0],
            [-jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1]
        ])
        coordinates.push_transform(transform)
