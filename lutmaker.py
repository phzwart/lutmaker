import math
import os
import sys
import random
from scipy.optimize import differential_evolution
import argparse

# Motor scale factors
x_scale_lib = {"501": -394, "502": 394, "503": -394}
t_scale_lib = {"501": -500, "502": -500, "503": -500}

# Coordinate frame conversions
x_factor = {"501": 1.0, "502": 1.0, "503": 1.0}
t_factor = {"501": 1.0, "502": 1.0, "503": 1.0}

# Moap coordinate pattern - A dictionary that stores the coordinates of pins in polar form
moap = {"11": (26.416, -163.636), "10": (26.416, -130.909), "13": (26.416, -229.090), "12": (26.416, -196.363),
        "15": (26.416, -294.545), "14": (26.416, -261.818), "16": (26.416, -327.272), "1": (12.19, 0.0),
        "3": (12.19, -144.0), "2": (12.19, -72.0), "5": (12.19, -288.0), "4": (12.19, -216.0),
        "7": (26.416, -32.727), "6": (26.416, 0.0), "9": (26.416, -98.181), "8": (26.416, -65.454),
        "c": (0.0, 0.0)}

def polar_to_cart(r, t):
    """
    Convert polar coordinates (r, theta) to cartesian coordinates (x, y).
    
    Parameters:
    r (float): The radial distance.
    t (float): The angle in degrees.
    
    Returns:
    (float, float): The cartesian coordinates (x, y).
    """
    x = r * math.cos(t * math.pi / 180.0)
    y = r * math.sin(t * math.pi / 180.0)
    return x, y

def nice_angle(t):
    """
    Normalize an angle to the range [-180, 180] degrees.
    
    Parameters:
    t (float): The angle in degrees.
    
    Returns:
    float: The normalized angle.
    """
    result = t % 360
    if result > 180:
        result = result - 360
    if result < -180:
        result = result + 360
    return result

class Puck:
    """
    Class representing a puck with pins and pattern.
    Handles conversion from polar coordinates to cartesian.
    """
    def __init__(self, pins=None, pattern=moap):
        """
        Initialize the puck with default pin configuration and pattern.
        
        Parameters:
        pins (list): A list of pin names (optional).
        pattern (dict): A dictionary containing pin coordinates in polar form.
        """
        if pins is None:
            pins = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "c"]
        self.pins = pins
        self.pattern = pattern
        self.xy = []
        self.com = [0, 0]
        self.theta = 0.0

    def set(self, com, theta):
        """
        Set the puck's center of mass and orientation.
        Converts the pin coordinates from polar to cartesian.
        
        Parameters:
        com (list): The center of mass coordinates [x, y].
        theta (float): The puck's orientation angle.
        """
        self.com = com
        self.theta = theta
        for pin in self.pins:
            r, t = self.pattern[pin]
            x, y = polar_to_cart(r, t + self.theta)
            self.xy.append([x + com[0], y + com[1]])

    def get_posi_in_robot_frame(self, cor):
        """
        Convert puck pin positions into the robot's coordinate frame.
        
        Parameters:
        cor (list): The robot's center of rotation [cx, cy].
        
        Returns:
        list: A list of [move, rotation] for each pin.
        """
        cx, cy = cor[0], cor[1]
        results = []
        r0 = math.sqrt(cx * cx + cy * cy)  # Radius to the origin from the center of rotation (cor)
        for xy, pid in zip(self.xy, self.pins):
            x, y = xy[0], xy[1]
            rp = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            xy1 = [x - cx, y - cy]
            xy2 = [0 - cx, 0 - cy]
            atan2_1 = math.atan2(xy1[1], xy1[0]) * 180.0 / math.pi
            atan2_2 = math.atan2(xy2[1], xy2[0]) * 180.0 / math.pi
            move = -abs(r0 - rp)
            rota = nice_angle(atan2_1 - atan2_2)
            results.append([move, rota])
        return results

class Fitter:
    """
    Class representing a fitter that optimizes puck positioning using differential evolution.
    """
    def __init__(self, id_data, data, thres=0.1):
        """
        Initialize the fitter with given id_data and observed data.
        
        Parameters:
        id_data (list): A list of pin IDs.
        data (list): The observed positions of the pins.
        thres (float): The optimization threshold (default: 0.1).
        """
        self.thres = thres
        self.id_data = id_data
        self.data = data
        self.x = None
        self.domain = [(-300, 307), (-307, 307), (-180, 180), (-200, -90), (-80, 10)]

    def target(self, vector):
        """
        The objective function to minimize during optimization.
        Computes the squared difference between observed and predicted positions.
        
        Parameters:
        vector (list): The vector representing [px, py, theta, cx, cy].
        
        Returns:
        float: The computed loss.
        """
        px, py, tp, cx, cy = vector
        td = Puck(self.id_data)
        td.set([px, py], tp)
        tmp = td.get_posi_in_robot_frame([cx, cy])
        res = 0
        for aa, bb in zip(self.data, tmp):
            res += (aa[0] - bb[0]) ** 2 + (aa[1] - bb[1]) ** 2
        return res

    def optimize(self):
        """
        Run the differential evolution optimization.
        
        Returns:
        result (OptimizeResult): The optimization result.
        """
        result = differential_evolution(self.target, self.domain, strategy="best1bin", mutation=(0.5, 1),
                                        popsize=50, disp=True, tol=1e-5, recombination=0.2, polish=True)
        self.x = result.x
        return result

    def return_puck(self):
        """
        Returns the puck pins and positions after optimization.
        
        Returns:
        tuple: A tuple containing puck pin IDs and their optimized positions.
        """
        if self.x is None:
            raise ValueError("Optimization has not been run yet.")
        
        px, py, tp, cx, cy = self.x
        td = Puck(self.id_data)
        td.set([px, py], tp)
        puck_positions = td.get_posi_in_robot_frame([cx, cy])
        return self.id_data, puck_positions

def parse_input_file(file_name, xfac=1.0, tfac=1.0):
    """
    Parse the input file containing puck data.
    
    Parameters:
    file_name (str): Path to the input file.
    xfac (float): Scaling factor for x coordinates.
    tfac (float): Scaling factor for theta (angle) values.
    
    Returns:
    tuple: A tuple containing the pin IDs and their locations.
    """
    with open(file_name, 'r') as file:
        pins = []
        locations = []
        for line in file:
            keys = line.split("\n")[0].split()
            pins.append(keys[0])
            locations.append([float(keys[1]) * xfac, nice_angle(float(keys[2])) * tfac])
    return pins, locations

def in_motor_positions(data, xscale, tscale, xfac, tfac):
    """
    Convert puck positions to motor positions.
    
    Parameters:
    data (list): A list of puck positions.
    xscale (float): Scale factor for x values.
    tscale (float): Scale factor for theta values.
    xfac (float): Scaling factor for x coordinates.
    tfac (float): Scaling factor for theta (angle) values.
    
    Returns:
    list: A list of motor positions.
    """
    result = []
    for dd in data:
        x = dd[0] * xfac
        t = dd[1] * tfac
        result.append([int(x * xscale + 0.5), int(t * tscale + 0.5)])
    return result

def get_single_puck(file_name, puck_name, xscale, tscale, xfac, tfac):
    """
    Process a single puck using differential evolution.
    
    Parameters:
    file_name (str): Path to the input file.
    puck_name (str): The puck name (e.g., 'A', 'B', etc.).
    xscale (float): Scale factor for x.
    tscale (float): Scale factor for theta.
    xfac (float): Scaling factor for x coordinates.
    tfac (float): Scaling factor for theta (angle) values.
    
    Returns:
    tuple: A tuple containing the puck names and motor positions.
    """
    obsid, data = parse_input_file(file_name, xfac, tfac)
    fitter_instance = Fitter(obsid, data)
    fitter_instance.optimize()  # Optimize first
    puck_pins, puck_positions = fitter_instance.return_puck()  # Return the puck results
    motor_positions = in_motor_positions(puck_positions, xscale, tscale, xfac, tfac)
    puck_names = [f"{puck_name}{pin}" for pin in puck_pins]
    
    print(f"\nPin number, predicted x, predicted theta (>>> deviation from observed)")
    for pin, (x_pred, theta_pred) in zip(puck_pins, puck_positions):
        print(f"{pin:5s} {x_pred:6.3f} {theta_pred:6.3f}")
    print(data)
 
    return puck_names, motor_positions

def process_pucks(params, xscale, tscale, xfac, tfac):
    """
    Process multiple pucks and fit them using the fitter.
    
    Parameters:
    params (argparse.Namespace): The parsed command-line arguments.
    xscale (float): Scale factor for x.
    tscale (float): Scale factor for theta.
    xfac (float): Scaling factor for x coordinates.
    tfac (float): Scaling factor for theta (angle) values.
    
    Returns:
    list: A list of results for each puck.
    """
    pucks = ['puckA', 'puckB', 'puckC', 'puckD', 'puckE', 'puckF', 'puckG', 'puckH', 'puckI', 'puckJ', 'puckK', 'puckL']
    results = []

    for puck in pucks:
        puck_path = getattr(params, puck, None)
        if puck_path:
            print(f"Working on {puck.upper()}")
            puck_res = get_single_puck(puck_path, puck.upper(), xscale, tscale, xfac, tfac)
            results.append(puck_res)

    return results

def main(args):
    """
    Main entry point for the puck fitter script.
    
    Parameters:
    args (list): Command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Puck Fitter")
    parser.add_argument('--site', choices=['501', '502', '503'], required=True, help="Site number")
    parser.add_argument('--puckA', type=str, help="Path to puckA data")
    parser.add_argument('--puckB', type=str, help="Path to puckB data")
    parser.add_argument('--puckC', type=str, help="Path to puckC data")
    parser.add_argument('--puckD', type=str, help="Path to puckD data")
    parser.add_argument('--puckE', type=str, help="Path to puckE data")
    parser.add_argument('--puckF', type=str, help="Path to puckF data")
    parser.add_argument('--puckG', type=str, help="Path to puckG data")
    parser.add_argument('--puckH', type=str, help="Path to puckH data")
    parser.add_argument('--puckI', type=str, help="Path to puckI data")
    parser.add_argument('--puckJ', type=str, help="Path to puckJ data")
    parser.add_argument('--puckK', type=str, help="Path to puckK data")
    parser.add_argument('--puckL', type=str, help="Path to puckL data")
    parser.add_argument('--output', type=str, help="Output file path")

    args = parser.parse_args(args)

    if not args.site:
        print("PLEASE SPECIFY WHICH ENDSTATION TO USE USING THE --site KEY")
        return

    # Scale factors
    xscale = x_scale_lib[args.site]
    tscale = t_scale_lib[args.site]
    xfac = x_factor[args.site]
    tfac = t_factor[args.site]

    # Process individual pucks
    puck_results = process_pucks(args, xscale, tscale, xfac, tfac)

    # Write output to a file if provided
    if args.output:
        with open(args.output, 'w') as f:
            for puck_name, puck_positions in puck_results:
                for name, pos in zip(puck_name, puck_positions):
                    f.write(f"{name} {pos[0]} {pos[1]}\n")
        print(f"Results written to {args.output}")
    else:
        # Print the results
        for puck_name, puck_positions in puck_results:
            for name, pos in zip(puck_name, puck_positions):
                print(f"{name}: {pos[0]}, {pos[1]}")

if __name__ == "__main__":
    main(sys.argv[1:])
