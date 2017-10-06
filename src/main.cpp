#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
     
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          //cout << "psi: " << psi << endl;
          double v = j[1]["speed"];
          //cout << "steering_angle: " << psi << endl;

          double delta = j[1]["steering_angle"];
          //cout << "steering_angle: " << psi << endl;
          double a = j[1]["throttle"];
          //cout << "throttle: " << psi << endl;

          //average accel (while throttle is 1) in m/s^2 calculated by vf^2 = vi^2 + 2*a*dist_travel
          //const double accel = 3.802128; 
          //double vel = v*0.44704; //convert velocity from mph to m/s

          // x,y, and psi wrt current car coordinate is always 0
          const double dt = 0.1;
          const double Lf = 2.67;

          // Recall the equations for the model:
          // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
          // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
          // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
          // v_[t+1] = v[t] + a[t] * dt
          // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
          // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
          // State after delay.

          // calculate everything based on delay
          /*
          px = px + vel*cos(psi)*dt;
          py = py + vel*sin(psi)*dt;
          psi = psi + (vel/Lf)*delta*dt;
          v = vel + accel*a*dt;
          */
          
          int Num_pts = ptsx.size();
          // 6 points
          //cout << "number of points to fit: " << Num_pts <<endl;
          Eigen::VectorXd ptsx_transformed(Num_pts);
          Eigen::VectorXd ptsy_transformed(Num_pts);

          // Transforms waypoints coordinates to the cars coordinates.
          for (int i = 0; i < Num_pts; i++ ) {
            double dx = ptsx[i] - px;
            double dy = ptsy[i] - py;
            double minus_psi = 0 - psi;
            ptsx_transformed[i] = dx * cos( minus_psi ) - dy * sin( minus_psi );
            ptsy_transformed[i] = dx * sin( minus_psi ) + dy * cos( minus_psi );
          }
          
          //third order fit to help smooth actuation
          auto coeffs = polyfit(ptsx_transformed,ptsy_transformed,3);

          // The cross track error is calculated by evaluating at polynomial at x, f(x)
          // and subtracting y.
          // f(x) = coeffs[0] + coeffs[1] * x  + coeffs[2] * x^2 + coeffs[3] * x^3 
          // y val at current x (wrt car, is 0) is just coeffs[0]
          double cte = coeffs[0];
          // Due to the sign starting at 0, the orientation error is -f'(x).
          // derivative of coeffs[0] + coeffs[1] * x  + coeffs[2] * x^2 + coeffs[3] * x^3 
          // -> coeffs[1] + 2 * coeffs[2] * x + 3 * coeffs[3] * x^2
          // again at current x (x = 0), derivative is just coeffs[1] at x = 0, psi = 0
          double epsi = -atan(coeffs[1]);
          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          const size_t N_States = 6;
          Eigen::VectorXd state(N_States);

          // Initial state.
          const double x0 = 0;
          const double y0 = 0;
          const double psi0 = 0;

          // Recall the equations for the model:
          // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
          // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
          // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
          // v_[t+1] = v[t] + a[t] * dt
          // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
          // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
          // State after delay.
          /*
          double x_delay = x0 + v*cos(psi0)*dt;
          double y_delay = y0 + v*sin(psi0)*dt;
          double psi_delay = psi0 + (v/Lf)*delta*dt;
          double v_delay = v + a*dt;
          double cte_delay = cte + v*sin(epsi)*dt;
          double epsi_delay = epsi + (v/Lf)*delta*dt;
          */
          //use delay state
          //state << x_delay, y_delay, psi_delay, v_delay, cte_delay, epsi_delay; 
          state << 0, 0, 0, v, cte, epsi; 

          auto mpc_solution = mpc.Solve(state, coeffs);

          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          double steer_value = -mpc_solution[0]/(deg2rad(25));
          double throttle_value = mpc_solution[1];

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          const size_t N = 10;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //move N-1 y trajectory
          for (int i = 0;i<N-1;i++) {
            mpc_x_vals.push_back(mpc_solution[2 + i]);
          }
          //cout << " x val " << mpc_x_vals.size() << endl;

          //move N-1 y trajectory
          for (int i = 0;i<N-1;i++) {
            mpc_y_vals.push_back(mpc_solution[2 + (N-1) + i]);
          }

          //cout << " y val " << mpc_y_vals.size() << endl;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          
          double step_size = 2.5;
          int steps = 10;
          for (int i = 1;i <= steps; i++) {
            next_x_vals.push_back(step_size*i);
            next_y_vals.push_back(polyeval(coeffs,step_size*i));
          }
          
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "\n42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
