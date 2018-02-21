import numpy

d=numpy.empty(2000)
print(d)
for i in range(100):
    d[i*20:i*20+10]=range(i*10,i*10+10)
    d[i*20+10:i*20+20]=range(1000+i*10,1000+i*10+10)
d=d.astype(numpy.int)
print(d)
print('D2')
d2=d.reshape(100, 10,2)
print(d2)
print(d2[0])