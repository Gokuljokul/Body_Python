x =int(input())
y1 = x % 1
x2 = x % 10
y2 = int(x2 - y1)
x3 = x % 100
y3 = int((x3 - x2) / 10)
x4 = x % 1000
y4 = int((x4 - x3) / 100)
x5 = x % 10000
y5 = int((x5 - x4) / 1000)
x6 = x % 100000
y6 = int((x6 - x5) / 10000)
x7 = x % 1000000
y7 = int((x7 - x6) / 100000)
x8 = x % 10000000
y8 = int((x8 - x7) / 1000000)
x9 = x % 100000000
y9 = int((x9 - x8) / 10000000)
x10 = x % 1000000000
y10 = int((x10 - x9) / 100000000)
x11 = x % 10000000000
y11 = int((x11 - x10) / 1000000000)

z = (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11)
print(z)

