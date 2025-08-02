#pragma once

class Equation {
private:
	int power;
public:
	Equation(int power);
	double calculate_term_value(double index, double x) const;
	int get_power();
};
