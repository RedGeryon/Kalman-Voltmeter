#***Kalman in 1D practice***
#Matrix object class is created by An Hoang (2015)
#This Kalman filter is a reproduction of Greg Czerniak's sample code (http://greg.czerniak.info/node/5),
#incorporating Udacity's ideas on Kalman Filtering in AI. Uses pylab (needs numpi and other related modules to graph)

#-----------------------------------------------------------------------------------------------------
#Matrix class and methods begin here
#-----------------------------------------------------------------------------------------------------
import random #we use this to add noise into our voltmeter measurements
import pylab #we use this later to make pretty graphs of our results (delete if not used)

class matrix(object):
    array = []
    row = 0
    col = 0
    
    def __init__(self, array, isFloat = False):
        #if [0][0] is float, rest is probably float (no map)
        self.isFloat = isFloat
        if (isFloat):
            self.array = array
        else:
            self.array = [[float(element) for element in row] for row in array]
            isFloat = True
        self.row = len(array)
        self.col = len(array[0])
    
    def __repr__(self):
        string = 'matrix\n['
        for i in range(self.row):
            string += (str(self.array[i]) + '\n')
        return string[:-1] +']'
    
    #helper func: epsilon rounding to prettify output
    def epsilon(self):
        mArr = self.array
        acc = 5 #accuracy to x significant figures
        ep = 0.0000001 #abitrary num, smaller than x sig figs
        return matrix([[round(num - ep, acc) for num in row] for row in mArr], True)

    #add matrices
    def __add__(self, other):
        assert (self.row == other.row and self.col == other.col), "Can only add matrices with same dimensions"
        return matrix([[i+j for i,j in zip(x,y)] for (x,y) in zip(self.array, other.array)], True)

    #subtract matrices
    def __sub__(self, other):
        assert (self.row == other.row and self.col == other.col), "Can only subtract matrices with same dimensions"
        return matrix([[i-j for i,j in zip(x,y)] for (x,y) in zip(self.array, other.array)], True)
    
    #multiply matrices or scalar with matrix      
    def __mul__(self, other):
        #handle scalar multiplication first
        if (type(other) in [int, float]):
            return matrix([[other * num for num in row] for row in self.array], True)
        assert (self.row == other.col), "Mismatch row and columns"
        #matrix * matrix
        product = []
        for rowcounter in range(self.row):
            singlerow = []
            for colcounter in range(other.col):
                cell = 0
                for i in range(other.row):
                    cell += (self.array[rowcounter][i] * other.array[i][colcounter])
                singlerow.append(cell)
            product.append(singlerow)
        return matrix(product, True)

    #help define for scalar on left side of multiplication
    #order of (matrix A * matrix B) wont be affected since it'll use top def
    __rmul__=__mul__
    
    #switch rows and columns (transpose a matrix)        
    def transpose(self):
        return matrix([list(i) for i in zip(*self.array)])
    
    #create identity matrix for square array
    def identity(self):
        assert (self.col == self.row), "Not a square matrix"
        I = [[1.0 if x == y else 0.0 for x in range(self.col)] for y in range(self.col)]
        return matrix(I, True)
    
    #find inverse of matrix via jordan-gauss elimination
    #apply complimentary operations to ID matrix to transforms
    #into inverse matrix (if inverse exists)
    def inverse(self):
        ID = matrix.identity(self).array
        mArr = self.array[:]
        
        #turn to echelon form:
        #checks to see if rows below first col is zeroed out
        #ifnot: sub scalar multiple of topmost echelon form row to zero out
        # c_ (subscript are the complimentary operations to ID matrix)
        def echelon(arr, comp):
            for row in range(1,self.row):
                for j in range(row, self.row):
                    if (arr[j][row - 1]):
                        scalar = (arr[j][row - 1]/arr[row-1][row-1])
                        temp = [scalar * a for a in arr[row - 1]]
                        arr[j] = [(a - b) for a, b in zip(arr[j], temp)]
                        c_temp = [scalar * a for a in comp[row - 1]]
                        comp[j] = [(a - b) for a, b in zip(comp[j], c_temp)]
            return arr, comp

        #scale diagonals to 1's
        def scale(arr, comp):
            for row in range(self.row):
                lead = arr[row][row]
                if (lead != 1):
                    scalar = 1/lead
                    arr[row] = [(scalar * a) for a in arr[row]]
                    comp[row] = [(scalar * a) for a in comp[row]]
            return arr, comp

        #turn rest into ID matrix
        def b_echelon(arr, comp):
            for row in range(self.row - 2, -1, -1):
                for j in range(self.row - 1, row, -1):
                    if (arr[row][j]):
                        scalar = (arr[row][j])
                        temp = [scalar * a for a in arr[j]]
                        arr[row] = [(a - b) for a, b in zip(arr[row], temp)]
                        c_temp = [scalar * a for a in comp[j]]
                        comp[row] = [(a - b) for a, b in zip(comp[row], c_temp)]
            return comp
        
        
        mArr, ID = echelon(mArr, ID)
        mArr, ID = scale(mArr, ID)
        ID = b_echelon(mArr, ID)

        #return after scaled through epsilon rounding
        return matrix(ID).epsilon()

#-----------------------------------------------------------------------------------------------------
#Matrix class ends here, and Kalman Filter Begins
#-----------------------------------------------------------------------------------------------------

class Kalman1D:
    def __init__(self, _A, _B, _H, _R, _Q, _xhat, _P):
        self.A = _A    # State transition matrix
        self.B = _B    # Control matrix
        self.H = _H    # State space/observation matrix
        self.R = _R    # Measurement error covariance
        self.Q = _Q    # Process/noise covariance
        self.curr_state_est = _xhat  # Initial state
        self.curr_covar_est = _P
    def GetState(self):
        return self.curr_state_est
    def Step(self, measurement, control_vector):
        #We input the control vector here so it can be modified with each step if needed
        #Prediction
        #--------------------------------------------------------------------------------
        predicted_state = self.A * self.curr_state_est + self.B * control_vector
        predicted_covariance = self.A * self.curr_state_est * A.transpose() + self.Q
        #Observation (Incorporate Sensor Data)
        #--------------------------------------------------------------------------------
        innovation = matrix([[measurement]]) - self.H * predicted_state     #Synonymous with y, innovation
        innovation_covariance = self.H * self.curr_covar_est * H.transpose() + self.R    #Synonymous with var S
        #Update
        #--------------------------------------------------------------------------------
        kalman_gain = self.curr_covar_est * self.H.transpose() * innovation_covariance.inverse() #Synonymous with var K
        self.curr_state_est = predicted_state + kalman_gain * innovation
        identity_matrix = self.curr_covar_est.identity()
        self.curr_covar_est = (identity_matrix - kalman_gain * self.H) * self.curr_covar_est

class Voltmeter:
    def __init__(self, _true_voltage, _noise):
        self.true_voltage = _true_voltage    #Set the 'true' voltage
        self.noise = _noise                  #Set noise to simulate Voltmeter variance via numpy Gaussian
    def trueVoltage(self):
        return self.true_voltage
    def measureVoltage(self):
        return random.gauss(self.trueVoltage(), self.noise)

#---------------------------------------------------------------------------------------
#Constants
#---------------------------------------------------------------------------------------
A = matrix([[1]])
B = matrix([[0]])
H = matrix([[1]])
R = matrix([[0.2]])
Q = matrix([[0.0001]])
xhat = matrix([[5]])
P = matrix([[1]])
u = matrix([[0]])       #control vector

num_of_steps = 50       #how many measurements we will take, in other circumstances will be timestep/timeslice
voltage = 1.50          #actual voltage of our system
noise = 0.3             #noise of our voltmeter

#---------------------------------------------------------------------------------------
#Actual measurements and Kalman filtering
#---------------------------------------------------------------------------------------

filter_step = Kalman1D(A, B, H, R, Q, xhat, P)      #initializing filter with our constants
voltmeter = Voltmeter(voltage, noise)               #initializing voltmeter with our values

actual = []     #actual voltage, will be constant
measured = []   #voltmeter measurement with noise
kalman = []     #incorporates measurement to guess

for steps in range(num_of_steps):
    voltmeter_reading = voltmeter.measureVoltage()
    measured.append(voltmeter_reading)
    actual.append(voltmeter.trueVoltage())
    kalman.append(filter_step.GetState().array[0][0])
    filter_step.Step(voltmeter_reading, u)

print(actual)
print('---------------------------------------------------------------------')
print(measured)
print('---------------------------------------------------------------------')
print(kalman)
#---------------------------------------------------------------------------------------
#Delete this part if you don't want to plot/don't have numpy/scipy
#---------------------------------------------------------------------------------------

pylab.plot(range(num_of_steps),measured,'b',range(num_of_steps),actual,'r',range(num_of_steps),kalman,'g')
pylab.xlabel('Steps')
pylab.ylabel('Voltage')
pylab.title('Kalman Filtered Est')
pylab.legend(('Measured','Actual Voltage','Kalman Est'))
pylab.show()
