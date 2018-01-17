#include "Halide.h"
#include <fstream>

using namespace Halide;
using std::vector;
using std::pair;

void thin_lens() {

    // Start by optimizing a thin lens to have the right focal length
    // using 2D ray transfer matrices. f is the focal length, g is the
    // distance between the lens and the sensor. The correct answer is
    // f == g.
    Param<float> p_f, p_g;
    Param<int> seed;

    Var x, u;

    // Wrap the params in Funcs, because we can't currently take derivatives w.r.t scalar params
    Func f, g;
    f() = p_f;
    g() = p_g;

    Func initial_u;
    initial_u(u) = random_float(seed) * 2 - 1;

    Func initial_x;
    initial_x(x, u) = random_float(seed) * 2 - 1;

    Func rays;
    rays(x, u) = {initial_x(x, u), initial_u(u)}; // rays(x, u) for fixed u is a 'packet' of parallel rays

    // Assume those rays are parameterized at the lens surface, so apply the lens
    Func lens;
    lens(x, u) = {rays(x, u)[0], rays(x, u)[1] - rays(x, u)[0] / f()};

    // Now add an air gap of size g to get the rays at the sensor
    Func sensor;
    sensor(x, u) = {lens(x, u)[0] + lens(x, u)[1] * g(), lens(x, u)[1]};

    // Require the rays to come to a focus. Use 10 rays per packet.
    RDom rx(0, 10);
    Func average_packet_x;
    average_packet_x(u) += sensor(rx, u)[0];
    average_packet_x(u) /= 10;

    Func packet_variance;
    packet_variance(u) += pow(sensor(rx, u)[0] - average_packet_x(u), 2);
    packet_variance(u) /= 9; // variance from samples, so 10-1

    Func loss;
    // Use 1024 packets
    RDom ru(0, 1024);
    loss() += packet_variance(ru);

    rays.compute_root();
    lens.compute_root();
    sensor.compute_root();
    average_packet_x.compute_root();
    packet_variance.compute_root();
    loss.compute_root();

    p_g.set(1);

    Buffer<float> l = loss.realize();
    printf("Loss = %f\n", l(0));

    auto d = Halide::propagate_adjoints(loss);

    // Optimize the focal length for an air gap of size 1.
    p_g.set(1);
    // The correct answer will be 1, because that's the definition of focal length.
    Func df = d(f);
    float focal_length = 3; // initial estimate
    for (int i = 0; i < 1000; i++) {
        seed.set(i); // Use a different random bundle of rays each time (SGD)
        p_f.set(focal_length);
        Buffer<float> dloss_df = df.realize();
        focal_length -= 0.001 * dloss_df();
    }

    if (fabs(focal_length - 1) > 0.0001) {
        printf("Incorrect focal length: %f\n", focal_length);
    } else {
        printf("Thin lens success!\n");
    }

}

// Returns the real part of the roots only
std::pair<Expr, Expr> solve_quadratic(Expr a, Expr b, Expr c) {
    Expr d = b*b - 4*a*c;
    float eps = 1e-10f;
    d = select(d <= 1e-10f, 0, sqrt(max(d, eps)));
    std::pair<Expr, Expr> roots = {(0 - b + d) / (2*a), (0 - b - d) / (2*a)};
    // Handle linear systems
    //roots.first = select(abs(a) < eps, -c / b, roots.first);
    //roots.second = select(abs(a) < eps, -c / b, roots.second);
    return roots;
}

void spherical_surface_2d() {
    // Try a lens with one spherical surface of the some radius r, in
    // 2D. The lens has index of refraction N. The sensor is inside
    // the lens (i.e. there's no second optical surface). If you
    // prefer you can imagine that the lens has a flat back and is
    // mounted directly on the sensor.

    Param<float> p_r, p_g, p_N;
    Param<int> seed;


    Var x, u;

    Func r, g, N;
    r() = p_r;
    g() = p_g;
    N() = p_N;

    Func initial_u;
    initial_u(u) = random_float(seed) * 2 - 1;

    Func initial_x;
    initial_x(x, u) = x/4.5f- 1; //random_float(seed) * 2 - 1;

    Func rays;
    rays(x, u) = {initial_x(x, u), initial_u(u)}; // rays(x, u) for fixed u is a 'packet' of parallel rays

    // Intersect with the first lens surface
    Func intersect1;
    {
        Expr X = rays(x, u)[0], U = rays(x, u)[1];
        Expr R = r();
        // X, U are the ray-space coordinates of ray x, u. To
        // intersect with the circle we propogate it forward by alpha,
        // which must satisfy:
        //
        // (X + alpha*U)^2 + (R - alpha)^2 = R^2
        //
        // Rearranging into a quadratic in alpha:
        //
        // (U^2 + 1) * alpha^2 + 2*(X*U - R) * alpha + X^2 = 0
        //
        // We want the closer intersection, so we solve for the smaller of the two roots of this quadratic
        Expr alpha = solve_quadratic(U*U + 1, 2*(X*U - R), X*X).second;
        X += alpha * U;

        // Compute the slope of the normal to the lens surface at the intersection
        Expr u3 = -X / (R - alpha);

        // Given an incoming slope u1, an outgoing slope u2, and the
        // slope of the normal ray at the intersection u3, Snell's law
        // painfully reformulated in terms of slopes and spreads
        // (squared sines from rational trigonometry) tells us:
        //
        // N^2 = [(u3 - u1)^2 * (1 + u2^2)] / [(u3 - u2)^2 * (1 + u1^2)]
        //
        // Expanding into a quadratic in u2:
        //
        // let A = N^2 * (1 + u1^2) in
        // let B = (u3 - u1)^2 in
        // A * (u3 - u2)^2 = B * (1 + u2^2)
        // =>
        // (A - B)*u2^2 - (2*A*u3)*u2 + (A*u3^2 - B) = 0
        Expr u1 = U;
        Expr A = N()*N() * (1 + u1*u1);
        Expr B = pow(u3 - u1, 2);
        auto roots = solve_quadratic(A - B, 0 - 2*A*u3, A*u3*u3 - B);
        // One of these roots corresponds to a negative index of
        // refraction N. We want the root that gives us a positive N,
        // which we get when u3 - u1 has the same sign as u3 -
        // u2. i.e. the slope before and after refraction shouldn't
        // hop over slope of the normal ray. If it's less than the
        // normal before refraction, it should be less than the normal
        // after refraction too.
        Expr u2 = select((u1 < u3) ^ (roots.first < u3), roots.second, roots.first);

        // Refract!
        U = u2;

        // We need our rays parameterized at a single consistent plane. Just walk them back to the original.
        X -= alpha * U;

        // X = print(X, "alpha =", alpha, "U =", U, "x =", x, "u =", u, "u3 =", u3, "roots", roots.first, roots.second, ") Refract:", rays(x, u)[0], rays(x, u)[1], " -> ", X, U);

        intersect1(x, u) = {X, U};

        intersect1.compute_root();
    }

    // Let's just take one surface for now.

    // Now add an air gap of size g to get the rays at the sensor
    Func sensor;
    {
        Expr X = intersect1(x, u)[0];
        Expr U = intersect1(x, u)[1];
        X += g() * U;
        // X = print(X, "x =", x, "u =", u, ") Propagate:", intersect1(x, u)[0], U, " -> ", X, U);
        sensor(x, u) = {X, U};
    }

    // Require the rays to come to a focus. Use 10 rays per packet.
    RDom rx(0, 10);
    Func average_packet_x;
    average_packet_x(u) += sensor(rx, u)[0];
    average_packet_x(u) /= 10;

    Func packet_variance;
    packet_variance(u) += pow(sensor(rx, u)[0] - average_packet_x(u), 2);
    packet_variance(u) /= 9; // variance from samples, so 10-1

    Func loss;
    // Use 1024 packets
    RDom ru(0, 1024);
    loss() += packet_variance(ru);
    loss() /= 1024;

    rays.compute_root();
    sensor.compute_root();
    average_packet_x.compute_root();
    packet_variance.compute_root();
    loss.compute_root();

    // Optimize the radius for an air gap of size 10, and an index of refraction 1.4
    p_g.set(10);
    p_r.set(10);
    p_N.set(1.4);

    Buffer<float> l = loss.realize();
    printf("Loss = %f\n", l(0));

    fprintf(stderr, "Calling propagate adjoints\n");
    auto d = Halide::propagate_adjoints(loss);

    Func dr = d(r);

    fprintf(stderr, "Entering optimization loop\n");
    float radius = 10; // initial estimate
    for (int j = 0; j < 50; j++) {
        for (int i = 0; i < 10; i++) {
            seed.set(j*10 + i); // Use a different random bundle of rays each time (SGD)
            p_r.set(radius);
            // fprintf(stderr, "radius: %f\n", radius);
            Buffer<float> dloss_dr = dr.realize();
            // fprintf(stderr, "dloss/dr: %f\n", dloss_dr());
            radius -= 0.5*dloss_dr();
        }

        loss.realize(l);
        printf("Radius of single spherical surface to achieve focal length = 1 is %f with error = %f\n",
               radius, l());
    }
}

std::ostream *stream = nullptr;
extern "C" void print_to_stream(void *user_context, const char *message) {
    if (stream) {
        (*stream) << message;
    }
}

void spherical_surface_2d(int n) {
    // Multiple spherical surfaces which alternate between air ->
    // glass and glass -> air.

    Buffer<float> radius_buf(n); // radius of each surface
    Buffer<float> z_buf(n); // distance between element i and element i+1 (element n is the image sensor).
    Buffer<float> N_buf(n); // Relative index of refraction of each surface.

    Param<int> seed;

    // We'll compile two pipeline variants - one for doing the
    // optimization, and one just for generating postscript figures.

    Pipeline p_optimize, p_render;

    for (int gen_postscript = 0; gen_postscript < 2; gen_postscript++) {

        const int rays_per_packet = 5;
        const int packets = 5;

        Var x, u;

        Func radius;
        radius(x) = radius_buf(x);

        Func z;
        z(x) = z_buf(x);

        Func N;
        N(x) = N_buf(x);

        Func initial_u;
        initial_u(u) = (0.2f*u)/(packets - 1) - 0.1f; //random_float(seed) * 2 - 1;

        Func initial_x;
        initial_x(x, u) = (2.0f*x)/(rays_per_packet - 1) - 1; //random_float(seed) * 2 - 1;

        Func rays;
        rays(x, u) = {initial_x(x, u), initial_u(u)}; // rays(x, u) for fixed u is a 'packet' of parallel rays

        vector<Func> funcs;
        funcs.push_back(rays);

        Expr current_z = 0.0f;
        for (int i = 0; i < n; i++) {
            // Intersect with the next lens surface
            Expr X = rays(x, u)[0], U = rays(x, u)[1];
            Expr R = radius(i);

            Expr X_initial = X;
            Expr Z_initial = current_z;

            // Walk the ray forward until it intersects the surface
            auto roots = solve_quadratic(U*U + 1, 2*(X*U - R), X*X);
            Expr alpha = select(R < 0, roots.first, roots.second);
            X += alpha * U;

            Expr X_at_lens_surface = X;
            Expr Z_at_lens_surface = Z_initial + alpha;

            // Compute the slope of the normal to the lens surface at the intersection
            Expr u3 = -X / (R - alpha);

            // Compute the slope of the outgoing ray
            Expr u1 = U;
            Expr A = N(i)*N(i) * (1 + u1*u1);
            Expr B = pow(u3 - u1, 2);
            roots = solve_quadratic(A - B, 0 - 2*A*u3, A*u3*u3 - B);
            Expr u2 = select((u1 < u3) ^ (roots.first < u3), roots.second, roots.first);

            // Refract!
            U = u2;

            // Walk the ray forwards to the plane of the next surface
            X += (z(i) - alpha) * U;


            if (gen_postscript) {

                Expr X_final = X;
                Expr Z_final = Z_initial + z(i);

                // If this is the first surface, start the rays a little early to make the figure clearer
                // Otherwise, go halfway back to the previous surface
                Expr dz = i == 0 ? 1 : z(i-1)/2;
                Z_initial -= dz;
                X_initial -= rays(x, u)[1] * dz;

                // Don't go all the way to the next surface, to avoid having to backtrack on concave surfaces
                if (i < n-1) {
                    Expr dz = z(i) / 2;
                    Z_final -= dz;
                    X_final -= U * dz;
                }

                // Emit the rays as postscript as a side-effect (Whee hack!)
                X = print(X, "pop", Z_initial, X_initial, "moveto",
                          Z_at_lens_surface, X_at_lens_surface, "lineto",
                          Z_final, X_final, "lineto",
                          "% PS", "seed =", seed, "x =", x, "u =", u);
            }


            Func next;
            next(x, u) = {X, U};
            rays = next;
            current_z += z(i);
            funcs.push_back(rays);
        }

        Func sensor = rays;

        // Require the rays to come to a focus.
        RDom rx(0, rays_per_packet);
        Func average_packet_x;
        average_packet_x(u) += sensor(rx, u)[0];
        average_packet_x(u) /= rays_per_packet;

        Func packet_variance;
        packet_variance(u) += pow(sensor(rx, u)[0] - average_packet_x(u), 2);
        packet_variance(u) /= rays_per_packet - 1; // variance from samples, so -1

        Func loss;
        RDom ru(0, packets);
        loss() += packet_variance(ru);
        loss() /= packets;

        // Constrain the thickness of the optical system
        loss() += 0.01f * pow(current_z - 10, 2);

        // Constrain the gaps to be positive
        for (int i = 0; i < n; i++) {
            loss() += 0.01f * pow(min(z(i), 0), 2);
        }

        funcs.push_back(average_packet_x);
        funcs.push_back(packet_variance);
        funcs.push_back(loss);

        // Initialize the system
        for (int i = 0; i < n; i++) {
            z_buf(i) = 5.0/n;
            N_buf(i) = (i % 2) ? (1/1.4f) : 1.4f;
            radius_buf(i) = 6.f; //(i >= n/2) ? -6.f : 6.f;
        }
        z_buf(n-1) = 10 - n;

        /*
          vector<vector<pair<int, int>>> bounds;
          bounds.push_back({{0, n-1}});
          bounds.push_back({{0, n-1}});
          vector<Func> outputs = p.outputs();
          simple_autoschedule(outputs, {}, bounds);
        */

        for (auto f : funcs) {
            f.compute_root();
        }

        if (gen_postscript) {
            p_render = Pipeline(loss);
            p_render.set_custom_print(print_to_stream);
        } else {
            auto d = Halide::propagate_adjoints(loss);

            for (auto f : funcs) {
                d(f).compute_root();
            }

            Func dr = d(radius);
            Func dz = d(z);

            p_optimize = Pipeline({dr, dz, loss});
        }
    }

    fprintf(stderr, "Entering optimization loop\n");
    Buffer<float> dloss_dr(n), dloss_dz(n);
    auto loss_buf = Buffer<float>::make_scalar();
    float learning_rate = 0.1f;
    for (int j = 0; j < 500; j++) {
        std::ofstream postscript("iter_" + std::to_string(j) + ".ps", std::ios_base::trunc);
        stream = &postscript;
        postscript << "<</PageSize [1200 800] >> setpagedevice\n"
                   << "100 400 translate\n"
                   << "100 100 scale\n"
                   << "0.01 setlinewidth\n";
        // Draw the lens surfaces
        float current_z = 0;
        for (int s = 0; s < n; s++) {
            float r = radius_buf(s);
            float theta = 180 * (asin(3 / std::max(3.01f, r)) / M_PI);
            bool front_surface = ((s & 1) == 0);
            if (front_surface) {
                postscript << "newpath ";
                theta = -theta;
            }
            postscript << (current_z + r) << " 0 " << r << " " << (180 - theta) << " " << (180 + theta);
            if (front_surface) {
                postscript << " arcn ";
            } else {
                postscript << " arc closepath 0.8 setgray fill 0.0 setgray stroke\n";
            }
            current_z += z_buf(s);
        }
        postscript << "newpath\n";
        // Draw the sensor
        postscript << current_z << " 4 moveto " << current_z << " -4 lineto\n";

        p_render.realize(loss_buf);

        postscript << "stroke showpage\n";
        stream = nullptr;
        postscript.close();

        float old_loss = loss_buf();
        for (int i = 0; i < 10000; i++) {
            seed.set(0);
            p_optimize.realize({dloss_dr, dloss_dz, loss_buf});
            for (int s = 0; s < n; s++) {
                radius_buf(s) -= learning_rate * dloss_dr(s);
                z_buf(s) -= 0.1f * learning_rate * dloss_dz(s);
            }
            if (loss_buf() < 0.995f * old_loss) break;
        }
        // Ramp up the learning rate slowly. It's not essential, but
        // it gives us more samples in time for the figures.
        if (learning_rate < 10) learning_rate *= 1.01;

        printf("Radii: ");
        for (int s = 0; s < n; s++) {
            printf("%f ", radius_buf(s));
        }
        printf("\nDistances: ");
        for (int s = 0; s < n; s++) {
            printf("%f ", z_buf(s));
        }
        printf("\nSpot size: %f\n", sqrt(loss_buf()));
        if (std::isnan(loss_buf())) break;
    }
}

int main(int argc, char **argv) {

    thin_lens();

    spherical_surface_2d();

    spherical_surface_2d(4);

    return 0;
}
