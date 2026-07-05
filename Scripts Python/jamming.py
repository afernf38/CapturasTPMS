python3 -c "
import sys, numpy as np
while True:
    # Portadora pura en formato int8 (lo que espera hackrf_transfer)
    samples = np.zeros(65536*2, dtype=np.int8)
    samples[0::2] = 120   # componente I
    samples[1::2] = 0     # componente Q
    sys.stdout.buffer.write(samples.tobytes())
    sys.stdout.buffer.flush()
" | hackrf_transfer -t - -f 433920000 -s 2000000 -x 47
