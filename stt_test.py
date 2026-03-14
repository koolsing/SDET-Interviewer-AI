import sys, select
i, o, e = select.select([sys.stdin], [], [], 0.1)
print(i)
