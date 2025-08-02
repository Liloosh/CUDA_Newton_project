#include "Equation.h"

Equation::Equation(int power): power(power) {
}

double Equation::calculate_term_value(double index, double x) const {
	double x_value = 1.0;
	for (int i = 0; i < power; i++) {
		x_value *= x;
	}
	return index * x_value;
}

int Equation::get_power() {
	return power;
}
