import numpy as np

# file for data

x_train = np.array([[(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)], 
                    [(1,1),(2,4),(3,9),(4,16),(5,25),(6,36)], 
                    [(0,1.5),(1,1.822),(2,2.203),(3,2.654),(4,3.193),(5,3.848)],[(1, 2.718), (2, 7.389), (3, 20.085), (4, 54.598), (5, 148.413), (6, 403.429)],
                    [(1, 6), (2, 15), (3, 28), (4, 45), (5, 66), (6, 91)],[(1, 1), (2, 8), (3, 27), (4, 64), (5, 125), (6, 216)],[(1, 3), (2, 9), (3, 27), (4, 81), (5, 243), (6, 729)],
                    [(1, 2.5), (2, 6.25), (3, 15.625), (4, 39.0625), (5, 97.65625), (6, 244.140625)],[(1, 2), (2, 5), (3, 10), (4, 17), (5, 26), (6, 37)],
                    [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12)],[(1, 3), (2, 6), (3, 9), (4, 12), (5, 15), (6, 18)],[(1, 3), (2, 7), (3, 11), (4, 15), (5, 19), (6, 23)],
                    [(1, 2.5), (2, 6.2), (3, 15.1), (4, 37.2), (5, 91.0), (6, 223.1)],[(1, 1.5), (2, 4.2), (3, 9.7), (4, 17.8), (5, 28.5), (6, 42.3)],
                    [(1, 1.5), (2, 4.2), (3, 9.7), (4, 17.8), (5, 28.5), (6, 42.3)],[(1, 1), (2, 3.2), (3, 5.4), (4, 7.6), (5, 9.8), (6, 12)],
                    [(1, 2.718), (2, 7.389), (3, 20.085), (4, 54.598), (5, 148.413), (6, 403.429)],[(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)],
                    [(2,4),(3,9),(4,16),(5,25),(6,36),(7,49)],[(1, 1.5), (2, 2.25), (3, 3.375), (4, 5.0625), (5, 7.59375), (6, 11.390625)],[(1, 3), (2, 9), (3, 27), (4, 81), (5, 243), (6, 729)],
                    [(10, 4), (11, 9), (12, 16), (13, 25),(14,36),(15,49)],[(-4, 0.5), (-3.5, 1.1), (-2.8, 2.3), (-1.6, 4.7), (-0.9, 9.2), (0.0, 15.0)],
                    [(-3.5, 1.1), (-2.8, 2.3), (-1.6, 4.7), (-0.9, 9.2), (0.0, 15.0), (1.2, 27.5)],[(-4, -7), (-3.5, -6.75), (-2.8, -5.6), (-1.6, -3.2), (-0.9, -1.8), (0.0, -1.0)],
                    [(-4, 10), (-3.5, 6.75), (-2.8, 4.68), (-1.6, 1.6), (-0.9, 0.21), (0.0, -0.5)],[(-4, 54.598), (-3.5, 33.115), (-2.8, 16.438), (-1.6, 5.332), (-0.9, 2.459), (0.0, 1.0)],
                    [(1, 4.0), (2, 2.5), (3, 1.7), (4, 1.1), (5, 0.7), (6, 0.5)],[(-6, 0.015625), (-5, 0.03125), (-4, 0.0625), (-3, 0.125), (-2, 0.25), (-1, 0.5)],
                    [(-3, 5), (-2, 8.5), (-1, 2.3), (0, 1.8), (1, 1), (2, 2.5)],[(5, 7), (6, 9.5), (7, 12), (8, 14.5), (9, 17), (10, 19.5)],[(5, 125), (6, 216), (7, 343), (8, 512), (9, 729), (10, 1000)],
                    [(10, 100), (11, 121), (12, 144), (13, 169), (14, 196), (15, 225)],[(-3, 47), (-2, 30), (-1, 13), (0, 2), (1, 3), (2, 14)],
                    [(-3, 47), (-1.5, 16.75), (0, 2), (1.5, 10.25), (3, 35), (4.5, 76.25)],[(-5, -16), (-2, -7), (0, -1), (3, 8), (6, 17), (8, 23)],
                    [(1.2, 2.4484), (4.8, 0.1477), (3.3, 0.3885), (7.1, 0.0278), (2.5, 0.6839), (6.4, 0.0541)],[(1.1, 5.5972), (2.3, 11.0365), (3.5, 21.7143), (4.7, 42.9508), (5.9, 84.5274), (7.1, 165.298)],
                    [(1,1),(2,8),(3,27),(4,64),(5,125),(6,216)],[(7.2, 373.248), (7.5, 421.875), (8.0, 512.0), (8.5, 614.125), (9.0, 729.0), (9.5, 857.375)],
                    [(0, 0.0), (1, 0.84), (2, 0.91), (3, 0.14), (4, -0.76), (5, -0.99)],[(1.2, 0.93), (2.5, 0.60), (3.7, -0.53), (5.1, -0.93), (6.4, 0.11), (7.6, 0.99)],
                    [(8, 0.99), (9.3, 0.10), (10.7, -0.98), (12, -0.54), (13.5, 0.39), (15, 0.65)],[(2, 0.91), (3, 0.14), (4, -0.76), (5, -0.99), (6, -0.28), (7, 0.66)],
                    [(-2, -0.91), (-3, -0.14), (-4, 0.76), (-5, 0.99), (-6, 0.28), (-7, -0.66)],[(-2, 4.09), (-3, 4.84), (-4, 4.37), (-5, 2.82), (-6, 1.14), (-7, 2.47)],
                    [(1, 4.61), (2, 4.99), (3, 2.91), (4, 0.92), (5, 1.99), (6, 3.92)],[(1, 2.38), (2, 2.97), (3, 1.71), (4, 0.48), (5, 1.54), (6, 2.81)],
                    [(-9, 0.93), (-8, 1.72), (-7, 3.61), (-6, 4.83), (-5, 4.09), (-4, 2.49)],[(8, 3.06), (9, 3.69), (10, 3.24), (11, 1.59), (12, 0.96), (13, 1.89)],
                    [(-3, 40), (-1, 7), (0, 2), (2, 1), (3, 4), (5, 82)],[(-3, -38), (-1, -2), (1, 4), (2, 16), (4, 86), (5, 163)],
                    [(0, 2), (2, 18), (3, 42), (5, 142), (7, 306), (9, 554)],[(1, 4), (3, 18), (5, 54), (7, 122), (9, 228), (11, 378)],
                    [(1, -2), (2, 5), (3, 32), (4, 109), (5, 278), (6, 575)],[(0, 1), (1, 0), (2, 1), (3, 4), (4, 9), (5, 16)],[(0, 0.0), (1, 1.68), (2, 2.95), (3, 3.61), (4, 3.54), (5, 2.81)],
                    [(0, 3.0), (1, 10.44), (2, 29.6), (3, 80.59), (4, 217.18), (5, 583.46)],[(0, 2.0), (1, 9.39), (2, 35.49), (3, 134.29), (4, 507.41), (5, 1922.11)],
                    [(10, 485165195.41), (11, 1318815734.48), (12, 3584912846.13), (13, 9744803446.25), (14, 26489122129.84), (15, 72004899337.39)],
                    [(5, 1096.63), (6, 2980.96), (7, 8103.08), (8, 22026.47), (9, 59874.14), (10, 162754.79)],[(5, 1100.25), (6, 2965.73), (7, 8115.67), (8, 21985.13), (9, 59890.42), (10, 162780.25)],
                    [(-1, 0.5), (0, 3.0), (1, 18.0), (2, 80.0), (3, 300.0), (4, 1100.0)],[(-3, 7.43), (-2, 3.08), (-1, 1.28), (0, 0.53), (1, 0.22), (2, 0.09)],[(0, 2.0), (1, 4.0), (2, 8.0), (3, 16.0), (4, 32.0), (5, 80.0)],
                    [(0, 2.0), (1, 4.0), (2, 15.0), (3, 16.0), (4, 32.0), (5, 64.0)],[(1, 3.0), (2, 15.0), (3, 12.0), (4, 24.0), (5, 48.0), (6, 96.0)],
                    [(0.5, 4.0), (1.5, 10.0), (2.5, 8.0), (3.5, 20.0), (4.5, 40.0), (5.5, 80.0)],[(1.2, 5.0), (3.4, 30.0), (2.0, 8.0), (4.5, 160.0), (0.8, 2.0), (5.1, 320.0)],
                    [(0.7, 3.0), (1.5, 8.0), (3.2, 25.0), (4.8, 85.0), (2.4, 12.0), (5.9, 210.0)],[(0.5, 10.0), (1.3, 5.0), (2.1, 2.0), (3.7, 0.5), (4.5, 0.3), (5.9, 0.1)],
                    [(1.2, 8.0), (2.8, 4.0), (3.5, 2.0), (4.1, 1.0), (5.7, 0.3), (6.4, 0.15)],[(0.1, 3.15), (1.2, 3.66), (2.3, 3.0), (3.4, 2.19), (4.5, 1.43), (5.6, 1.0)],[(0.2, 2.87), (1.5, 1.48), (2.8, 0.31), (4.1, 0.94), (5.4, 2.17), (6.7, 3.0)],
                    [(0.3, 1.96), (1.6, 2.97), (2.9, 2.23), (4.2, 0.87), (5.5, 0.12), (6.8, 0.95)],[(0.0, 0.0), (1.3, 1.92), (2.6, 2.87), (3.9, 1.73), (5.2, 0.69), (6.5, 0.04)],
                    [(0.0, 0.0), (1.5, 1.83), (3.0, 2.93), (4.5, 1.95), (6.0, 0.27), (7.5, -0.98)],[(-1.0, -0.84), (0.5, 1.23), (2.0, 2.91), (3.5, 1.84), (5.0, -0.17), (6.5, -1.5)],
                    [(-2.0, -1.1), (-0.5, 0.87), (1.3, 2.68), (3.7, 1.24), (5.5, -0.61), (7.8, -1.89)],[(-1.5, -0.98), (-0.3, 0.63), (1.7, 2.56), (3.2, 1.98), (5.0, 0.13), (6.8, -1.14)],
                    [(-2.5, -1.57), (-1.0, -0.84), (0.8, 0.97), (2.3, 2.72), (4.1, 0.96), (6.0, -0.92)],[(70.0, 0.64), (71.5, -0.97), (73.0, -1.82), (75.0, -0.38), (77.0, 1.68), (78.5, 2.91)],
                    [(4.0, -1.7), (5.0, 4.8), (6.0, 7.9), (7.0, 2.3), (8.0, -6.5), (9.0, -9.2)],[(8.0, -3.2), (9.0, 7.5), (10.0, 9.8), (11.0, -4.1), (12.0, -11.5), (13.0, 5.7)],
                    [(3.0, 20.1), (4.0, 10.5), (5.0, -5.2), (6.0, -17.8), (7.0, 5.6), (8.0, 23.0)],[(2.5, 15.2), (3.5, 5.6), (4.5, -7.3), (5.5, -19.8), (6.5, 2.4), (7.5, 20.5)],
                    [(1.0, 3.2), (2.0, 1.8), (3.0, -1.5), (4.0, -3.9), (5.0, 0.7), (6.0, 4.1)],[(0.0, 4.0), (1.0, 4.55), (2.0, 4.32), (3.0, 2.84), (4.0, 1.32), (5.0, 2.32)],
                    [(0.0, 5.5), (1.0, 4.76), (2.0, 1.96), (3.0, 0.62), (4.0, 2.06), (5.0, 4.74)],[(0.0, 0.0), (0.698, 0.644), (1.396, 0.992), (2.094, 0.866), (2.793, 0.342), (3.491, -0.342)],
                    [(0.0, 0.0), (0.785, 0.707), (1.571, 1.0), (2.356, 0.707), (3.142, 0.0), (3.927, -0.707)],[(0.0, 4.0), (0.6, 2.22), (1.2, 1.54), (1.8, 1.27), (2.4, 1.11), (3.0, 1.07)],
                    [(-4.0, 13.63), (-3.0, 9.41), (-2.0, 6.5), (-1.0, 4.78), (0.0, 5.0), (1.0, 4.82)],[(-4.0, 30.65), (-2.8, 15.45), (-1.6, 8.78), (-0.4, 5.17), (0.8, 3.52), (2.0, 2.5)],
                    [(-2.0, 35.0), (-1.0, 8.0), (0.0, 1.0), (1.0, -2.0), (2.0, 3.0), (3.0, 46.0)],[(-4.0, 2.7), (-3.0, 4.2), (-2.0, 5.6), (-1.0, 7.2), (0.0, 8.1), (1.0, 8.9)],
                    [(-2.0, 4.39), (-1.5, 6.01), (-1.0, 6.89), (-0.5, 7.5), (0.0, 7.97), (0.5, 8.35)],[(1, 0), (2, 0.69), (3, 1.09), (4, 1.38), (5, 1.61), (6, 1.79)],
                    [(7, 11.95), (7.2, 11.97), (8, 12.07), (10, 12.3), (11, 12.39), (15, 12.71)],[(-2.87, 16.55), (-2.16, 20.64), (-2.11, 20.85), (0.07, 26.27), (3.18, 29.96), (4.83, 31.26)],
                    [(-2.0, 24.08), (-1.58, 25.42), (1.7, 32.04), (3.28, 34.06), (3.59, 34.4), (4.2, 35.05)],[(-3.77, 3.98), (-2.43, 4.85), (1.17, 6.19), (3.41, 6.72), (3.76, 6.8), (4.54, 6.95)],
                    [(-3.57, 2.53), (-3.56, 2.57), (-1.81, 4.64), (0.16, 5.33), (0.44, 5.39), (1.04, 5.53)],[(-2.51, 15.3), (-0.19, 23.76), (3.37, 28.87), (4.19, 29.66), (6.01, 31.15), (8.09, 32.53)],
                    [(-2.89, 12.57), (-1.38, 20.96), (-0.9, 21.94), (2.87, 25.94), (4.23, 26.76), (4.76, 27.04)],[(0.84, 33.97), (3.2, 38.62), (3.56, 39.12), (5.23, 41.06), (9.19, 44.24), (9.2, 44.24)],
                    [(3.42, 4.69), (5.04, 5.03), (6.83, 5.31), (7.68, 5.42), (8.75, 5.54), (9.82, 5.65)],[(0.95, 33.33), (2.05, 40.58), (2.08, 40.73), (3.1, 44.9), (4.72, 49.49), (8.63, 56.33)],
                    [(-5.16, 17.61), (-3.28, 21.95), (-1.99, 23.45), (-1.49, 23.91), (2.57, 26.43), (5.69, 27.66)],[(-2.24, 16.41), (-1.14, 19.87), (0.45, 22.59), (5.48, 26.82), (7.85, 28.01), (9.67, 28.76)],
                    [(7.03, 36.34), (8.26, 37.39), (8.36, 37.47), (11.1, 39.34), (16.79, 42.11), (19.48, 43.12)],[(8.44, 44.6), (8.68, 44.8), (9.16, 45.18), (11.47, 46.82), (16.59, 49.57), (16.98, 49.75)],
                    [(3.1, 26.22), (9.2, 31.68), (10.77, 32.54), (11.58, 32.94), (11.94, 33.11), (15.3, 34.5)],[(5.68, 20.7), (5.95, 20.84), (7.0, 21.33), (15.6, 24.02), (17.26, 24.38), (19.74, 24.86)],
                    [(1.12, 35.21), (1.17, 35.43), (1.21, 35.6), (1.45, 36.54), (1.58, 37.01), (1.84, 37.88)],[(79.93, 25.35), (114.51, 26.42), (175.27, 27.7), (194.5, 28.01), (308.37, 29.39), (367.65, 29.92)],
                    [(3.35, 50.44), (3.49, 50.82), (4.36, 52.94), (5.25, 54.76), (6.25, 56.5), (7.75, 58.67)],[(0.15, 1.14), (1.61, 1.96), (2.47, 2.24), (2.9, 2.36), (3.24, 2.44), (3.53, 2.51)],
                    [(3.5, 2.5), (3.65, 2.54), (3.97, 2.6), (6.17, 2.97), (6.19, 2.97), (7.93, 3.19)],[(0.27, 1.24), (1.57, 1.94), (2.74, 2.32), (4.67, 2.74), (5.71, 2.9), (8.18, 3.22)],
                    [(0.23, 1.21), (3.34, 2.47), (3.72, 2.55), (3.92, 2.59), (6.71, 3.04), (8.36, 3.24)],[(0.63, 1.49), (1.16, 1.77), (3.05, 2.4), (3.23, 2.44), (4.53, 2.71), (5.63, 2.89)],
                    [(2, 2.1), (3, 2.39), (4, 2.61), (5, 2.79), (6, 2.95), (7, 3.08)],])


y_train = np.array([["polynomial"], ["polynomial"], ["exponential"],["exponential"],["polynomial"],["polynomial"],["exponential"],["exponential"],["polynomial"],
                    ["polynomial"],["polynomial"], ["polynomial"],["exponential"],["polynomial"],["polynomial"],["polynomial"],["exponential"],["polynomial"],["polynomial"],
                    ["exponential"],["exponential"],["polynomial"],["exponential"],["exponential"],["polynomial"],["polynomial"],["exponential"],["exponential"],
                    ["exponential"],["polynomial"],["polynomial"],["polynomial"],["polynomial"],["polynomial"],["polynomial"],["polynomial"],["exponential"],["exponential"],
                    ["polynomial"],["polynomial"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["polynomial"],
                    ["polynomial"],["polynomial"],["polynomial"],["polynomial"],["polynomial"],["sine"],["exponential"],["exponential"],["exponential"],["exponential"],
                    ["exponential"],["exponential"],["exponential"],["exponential"],["exponential"],["exponential"],["exponential"],["exponential"],["exponential"],["exponential"],["exponential"],
                    ["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],["sine"],
                    ["exponential"],["exponential"],["exponential"],["polynomial"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],
                    ["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"],["ln"]])


