if __name__ == '__main__':
    from detector import Detector
    import time
    import numpy as np
    N = 100
    model = Detector()
    
    t0 = time.perf_counter()
    for _ in range(N):
        t = (np.random.uniform(size=(480, 480, 3)) * 255).astype(np.uint8)        
        model.predict(t)
    tn = time.perf_counter()
    
    print(f"Time taken for {N} iterations: {tn - t0:.2f} seconds")
        
    
