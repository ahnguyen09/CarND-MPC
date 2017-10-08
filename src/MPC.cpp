#include "MPC.h"
#include <math.h>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
// number to timestep to evaluate in into the future
const size_t N = 12;

const size_t N_States = 6;
const size_t N_Actuators = 2;

// time
const double dt = 0.1;
// I started with dt = 0.1 to match delay and N = 20.
// With N = 20, the calculated trajectory was following the yellow line for too long.
// This means that cost contribution after a certain N steps is negligible and a waste in computation.
// So I decrease N to allow calculated trajectory to get to the yellow line. I feel like it looks good at N=12 and dt = 0.1.

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;
//average accel (while throttle is 1) in m/s^2 calculated by vf^2 = vi^2 + 2*a*dist_travel
const double accel = 3.802128; 

// The solver takes all the state variables and actuator
// variables in a singular vector. Keep track of starting vector position


// size of these vector are N - 1
const size_t delta_start = 0;
const size_t a_start = delta_start + (N-1);

// weights for cost computations
const double W_cte = 5.0;
const double W_epsi = 25.0; // we care more about epsi if we want to limit over steer
const double W_v = 5.0;

const double W_delta = 4000.0; // punish delta heavily to prevent large oscilation/ over steer
const double W_a = 5.0;

const double W_delta_diff = 100.0; 
const double W_a_diff = 5.0; // 
typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

// Evaluate a polynomial.
AD<double> ADpolyeval(Eigen::VectorXd coeffs, AD<double> x) {
  AD<double> result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Evaluate derivative polynomial.
AD<double> ADpolyevalDiff(Eigen::VectorXd coeffs, AD<double> x) {
  AD<double> result = 0.0;
  for (int i = 1; i < coeffs.size(); i++) {
    result += i*coeffs[i] * pow(x, i-1);
  }
  return result;
}

// Based on simplifying MPC forum post
// https://discussions.udacity.com/t/simplyfied-state-in-mpc/382849
// https://github.com/pkol/CarND-MPC-Quizzes/blob/master/mpc_to_line/src/MPC.cpp

class FG_eval {
 public:
  // Fitted polynomial coefficients
  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs, vector<double> init_state) {
    this->coeffs = coeffs;
    this->x0 = init_state[0];
    this->y0 = init_state[1];
    this->psi0 = init_state[2];
    this->v0 = init_state[3];
    this->cte0 = init_state[4];
    this->epsi0 = init_state[5];
  }

  double x0, y0, psi0, v0, cte0, epsi0;

  void run_car_model(const ADvector& vars, ADvector& x_out, ADvector& y_out, ADvector& psi_out, ADvector& v_out,
                     ADvector& cte_out, ADvector& epsi_out)
  {
    // vars (actuation) will be what we will solve for using Ipopt
    // however we should complete car model

    // make sure to pass in m/s not mph

    //set initial state [t=0]
    x_out[0] = x0;
    y_out[0] = y0;
    psi_out[0] = psi0;
    v_out[0] = v0;
    cte_out[0] = cte0;
    epsi_out[0] = epsi0;

    //calculate
    for(size_t t=1; t< N; t++){
      AD<double> x_i = x_out[t-1];
      AD<double> y_i = y_out[t-1];
      AD<double> psi_i = psi_out[t-1];
      AD<double> v_i = v_out[t-1];
      AD<double> cte_i = cte_out[t-1];
      AD<double> epsi_i = epsi_out[t-1];

      auto delta_i = vars[delta_start + t - 1]; //heading (radians)
      auto a_i = vars[a_start + t - 1]; //throttle between -1,1

      // Recall the equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      x_out[t] = x_i + v_i * CppAD::cos(psi_i) * dt;
      y_out[t] = y_i + v_i * CppAD::sin(psi_i) * dt;
      psi_out[t] = psi_i + (v_i/Lf)*delta_i*dt;
      v_out[t] = v_i + a_i*accel*dt; //scale accel based on throttle
      
      cte_out[t] = ADpolyeval(coeffs,x_i) - y_i + v_i*CppAD::sin(epsi_i)*dt;
      AD<double> psides_i = CppAD::atan(ADpolyevalDiff(coeffs,x_i));
      epsi_out[t] = psi_i - psides_i +  (v_i/Lf)*delta_i*dt;   
    }
  }

  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    ADvector x_pred(N), y_pred(N), psi_pred(N), v_pred(N), cte_pred(N), epsi_pred(N);
    run_car_model(vars, x_pred, y_pred, psi_pred, v_pred,cte_pred,epsi_pred);

    //reset fg[0] (cost) before accumulating 
    fg[0] = 0;

    //in terms of m/s!
    double ref_v = 33.528; // 75 mph

    //accumulate the trajectory error
    for (size_t t = 0; t < N; t++) {
      fg[0] += W_cte * CppAD::pow(cte_pred[t], 2); //cross-track error squared
      fg[0] += W_epsi * CppAD::pow(epsi_pred[t], 2); //orientation error
      fg[0] += W_v * CppAD::pow(ref_v - v_pred[t], 2); //velocity error squared
    }

    // Minimize the use of actuators.
    for (size_t t = 0; t < N - 1; t++) {
      fg[0] += W_delta * CppAD::pow(vars[delta_start + t], 2); //steering angel
      fg[0] += W_a * CppAD::pow(vars[a_start + t], 2); // throttle
    }

    // Minimize the value gap between sequential actuations.
    // help prevent passengers from getting motion sickness
    for (size_t t = 0; t < N - 2; t++) {
      fg[0] += W_delta_diff * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2); // turning rate
      fg[0] += W_a_diff * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2); // throttle rate
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  //size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  vector<double> init_state(N_States);

  init_state[0] = state[0]; //x
  init_state[1] = state[1]; //y
  init_state[2] = state[2]; //psi
  init_state[3] = state[3]; //v - in m/s
  init_state[4] = state[4]; //cte
  init_state[5] = state[5]; //epsi

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is
  size_t n_vars = N_Actuators*(N-1);
  // TODO: Set the number of constraints
  size_t n_constraints = 0;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (size_t i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  for (size_t i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // throttle upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (size_t i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }
  // TODO: Set lower and upper limits for variables.

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.

  //empty list?
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (size_t i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs,init_state);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  FG_eval::ADvector vars_ster(n_vars), x_pred(N), y_pred(N), psi_pred(N), v_pred(N), cte_pred(N), epsi_pred(N);
  for(size_t i=0; i<n_vars; i++) vars_ster[i] = solution.x[i];
  fg_eval.run_car_model(vars_ster, x_pred, y_pred, psi_pred, v_pred,cte_pred,epsi_pred);

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  vector<double> result;
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  //move N-1 x trajectory
  for (size_t t = 1;t<N;t++) {
    result.push_back(CppAD::Value(x_pred[t]));
  }
  //move N-1 y trajectory
  for (size_t t = 1;t<N;t++) {
    result.push_back(CppAD::Value(y_pred[t]));
  }
  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.

  // steer(delta), throttle, (a), x_next, y_next
  return result;
}
