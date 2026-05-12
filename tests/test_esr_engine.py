import unittest
import numpy as np
from src.esr_engine import ESREngine
from src.config import PipelineConfig

class TestESREngine(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig(env="local")
        self.engine = ESREngine(self.config)

    def test_lipschitz_stability_stable(self):
        # Create a smooth landscape
        Z_grid = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                Z_grid[i, j] = 0.1 * i + 0.1 * j
        
        # Should not raise an error
        self.assertTrue(self.engine.check_lipschitz_stability(Z_grid, threshold=0.5))

    def test_lipschitz_stability_unstable(self):
        # Create a chaotic landscape
        Z_grid = np.random.rand(5, 5)
        
        # Should raise ValueError for high threshold
        with self.assertRaises(ValueError):
            self.engine.check_lipschitz_stability(Z_grid, threshold=0.01)

    def test_dynamic_boundaries(self):
        # Test if boundaries are generated correctly
        X_clean = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        N, D = X_clean.shape
        
        X_g, Y_g, Z_g = self.engine.map_surrogate_landscape(X_clean, y, N, D)
        
        self.assertEqual(X_g.shape, (5, 5))
        self.assertEqual(Y_g.shape, (5, 5))
        self.assertEqual(Z_g.shape, (5, 5))
        
        # Check if max_depth_upper is calculated correctly
        max_depth_upper = min(30, int(np.log2(N) * 2))
        self.assertAlmostEqual(X_g.max(), max_depth_upper)

if __name__ == '__main__':
    unittest.main()
