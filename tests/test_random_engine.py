import DistGNN
import dgs
import time

begin = time.time()
for i in range(10):
    print(dgs.ops._CAPI_Randn())
print(time.time() - begin)