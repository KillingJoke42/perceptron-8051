#import tensorflow as tf
import numpy as np

def relu(x):
	return int(x * (x > 0))

def optimizer(w, b, x, y, lr, num_iterations):
	while(num_iterations):
		fwd = relu(np.add(np.dot(w,x), b))
		dw = np.multiply(x, (fwd - y))
		print("epoch: {}; cost: {}".format(num_iterations, np.sum(np.power(fwd - y, 2))))
		db = fwd - y
		w = round(w - lr*dw, 3)
		b = round(b - lr*db, 3)
		num_iterations -= 1

	return w, b

def train(X_train, Y_train, epochs=10000, lr=0.001, w=np.random.rand(), b=0):
	w, b = optimizer(w, b, X_train, Y_train, lr, epochs)
	with open("keil_files/weights.c", "w") as wgtsfile:
		wgtsfile.write("#include \"weights.h\"\n")
		wgtsfile.write("float weight(void) {return " + str(w) + ";}\n")
		wgtsfile.write("float bias(void) {return " + str(b) + ";}\n")
	return 0

def main():
	train(1, 255)

if __name__ == '__main__':
	main()