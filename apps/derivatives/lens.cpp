#include "Halide.h"
#include <fstream>

using namespace Halide;
using std::vector;
using std::pair;

// Solve a quadratic. Returns the real part of the roots only
std::pair<Expr, Expr> solve_quadratic(Expr a, Expr b, Expr c) {
    Expr d = b*b - 4*a*c;
    float eps = 1e-20f; // Avoid infinite derivatives at zero.
    d = select(d <= eps, 0, sqrt(max(d, eps)));
    std::pair<Expr, Expr> roots = {(0 - b + d) / (2*a), (0 - b - d) / (2*a)};
    return roots;
}

// Trace rays through a system of spherical lenses, optimizing the lens parameters.
void spherical_lens_system_2d() {
    int n; // number of surfaces
    float fov; // The maximum ray slope at the aperture. A measure of field of view.
    float F; // Proportional to the extent of the ray packet at the aperture. The F-number.
    Buffer<float> radius_buf; // radius of each surface
    Buffer<float> z_buf;      // distance between element i and element i+1 (element n is the image sensor).
    Buffer<float> N_buf;      // Relative index of refraction of each surface, at 3 different wavelengths.
    Param<float> sensor_z;    // The depth of the image sensor

    // Initialize the system
    {
        // Zeiss 1955 design from US2799207A
        // Constants from the patent:
        float
            r1 = 0.347219, r2 = 0.904131, r3 = 0.309462, r4 = 0.215955,
            r5 = -0.228743, r6 = 1.69089, r7 = -0.320794, r8 = -2.83857,
            r9 = -0.538549, d1 = 0.07573, l1 = 0.00132, d2 = 0.06236,
            l2 = 0.24492, d3 = 0.01324, d4 = 0.09492, l3 = 0.00132,
            d5 = 0.06620, n1 = 1.62041, v1 = 60.3, n2 = 1.75520,
            v2 = 27.5, n3 = 1.71736, v3 = 29.5, n4 = 1.69067,
            v4 = 54.9, n5 = 1.75520, v5 = 27.5;

        n = 9;
        radius_buf = Buffer<float>(n);
        z_buf = Buffer<float>(n);
        N_buf = Buffer<float>(n, 3);

        // Scale up the focal length to 10.
        float scale = 7.715395f;

        radius_buf(0) = r1 * scale;
        radius_buf(1) = r2 * scale;
        radius_buf(2) = r3 * scale;
        radius_buf(3) = r4 * scale;
        radius_buf(4) = r5 * scale;
        radius_buf(5) = r6 * scale;
        radius_buf(6) = r7 * scale;
        radius_buf(7) = r8 * scale;
        radius_buf(8) = r9 * scale;

        z_buf(0) = 0;
        z_buf(1) = d1 * scale;
        z_buf(2) = l1 * scale;
        z_buf(3) = d2 * scale;
        z_buf(4) = l2 * scale;
        z_buf(5) = d3 * scale;
        z_buf(6) = d4 * scale;
        z_buf(7) = l3 * scale;
        z_buf(8) = d5 * scale;

        // Integrate to get absolute coordinates of each element's intersection with the optical axis
        for (int i = 1; i < 9; i++) {
            z_buf(i) += z_buf(i-1);
        }

        // If we assume that for the material, nD = (nF + nC)/2, the
        // Abbe number and nD are sufficient to calculate an index of
        // refraction at all three wavelengths.
        float dn1 = (n1 - 1) / (2 * v1);
        float dn2 = (n2 - 1) / (2 * v2);
        float dn3 = (n3 - 1) / (2 * v3);
        float dn4 = (n4 - 1) / (2 * v4);
        float dn5 = (n5 - 1) / (2 * v5);

        // Absolute indices of refraction
        N_buf.fill(1.0); // Initialize to air
        N_buf(0, 0) = n1 - dn1;
        N_buf(0, 1) = n1;
        N_buf(0, 2) = n1 + dn1;
        N_buf(2, 0) = n2 - dn2;
        N_buf(2, 1) = n2;
        N_buf(2, 2) = n2 + dn2;
        N_buf(4, 0) = n3 - dn3;
        N_buf(4, 1) = n3;
        N_buf(4, 2) = n3 + dn3;
        N_buf(5, 0) = n4 - dn4;
        N_buf(5, 1) = n4;
        N_buf(5, 2) = n4 + dn4;
        N_buf(7, 0) = n5 - dn5;
        N_buf(7, 1) = n5;
        N_buf(7, 2) = n5 + dn5;

        // Convert to relative indices of refraction at each optical surface.
        for (int i = n-1; i > 0; i--) {
            for (int j = 0; j < 3; j++) {
                N_buf(i, j) /= N_buf(i-1, j);
            }
        }

        fov = 1/6.0;
        F = 6.0;
    }

    // We'll compile two pipeline variants - one computing loss and
    // gradient, and one just for evaluating loss to be used during
    // line search.

    Pipeline p_optimize, p_evaluate;
    {
        const int rays_per_packet = 8;
        const int packets = 8;

        Var x, u, l;

        // We take derivatives w.r.t. Funcs or ImageParams, not raw
        // Buffers, so wrap the input buffers in a Func.
        Func radius;
        radius(x) = radius_buf(x);

        Func z;
        z(x) = z_buf(x);

        Func N;
        N(x, l) = N_buf(x, l);

        Func initial_u;
        initial_u(u) = ((cast<float>(2*u))/(packets - 1) - 1) * fov;

        float focal_length = 10; // distance from aperture to sensor
        float aperture_diameter = focal_length / F;
        float aperture_radius = aperture_diameter / 2;
        Func initial_x;
        initial_x(x) = (cast<float>(2*x)/(rays_per_packet - 1) - 1) * aperture_radius;

        // rays(x, u, l) is a ray of intial height above the optical
        // axis x, initial slope u, and wavelength l. The three Tuple
        // components are the x coord of the ray, the u coord, and z.
        Func rays;
        rays(x, u, l) = {initial_x(x), initial_u(u), 0.f};

        vector<Func> funcs;
        rays.vectorize(x);
        funcs.push_back(rays);

        Func loss;
        loss() = 0.f;

        RDom rr(0, rays_per_packet, 0, packets, 0, 3);

        for (int i = 0; i < n; i++) {
            // Intersect with the next lens surface
            Expr X = rays(x, u, l)[0], U = rays(x, u, l)[1], Z = rays(x, u, l)[2];
            Expr R = radius(i);

            Expr X_initial = X;
            Expr Z_initial = Z;

            // Walk the ray to the reference plane for this surface
            X += (z(i) - Z) * U;
            Z = z(i);

            // Walk the ray forward (or backward) some more until it intersects the surface
            auto roots = solve_quadratic(U*U + 1, 2*(X*U - R), X*X);
            Expr alpha = select(R < 0, roots.first, roots.second);
            X += alpha * U;
            Z += alpha;

            // Compute the slope of the normal to the lens surface at the intersection
            Expr u3 = -X / (R - alpha);

            // Compute the slope of the outgoing ray using Snell's law
            // expressed in Cartesian ray-space.
            Expr u1 = U;
            Expr A = N(i, l)*N(i, l) * (1 + u1*u1);
            Expr B = pow(u3 - u1, 2);
            roots = solve_quadratic(A - B, 0 - 2*A*u3, A*u3*u3 - B);
            Expr u2 = select((u1 < u3) ^ (roots.first < u3), roots.second, roots.first);

            // Refract!
            U = u2;

            Func next;
            next(x, u, l) = {X, U, Z};

            // Force each ray to travel a minimum length, to avoid infinitely thin lenses.
            loss() += pow(min(next(rr.x, rr.y, rr.z)[2] - rays(rr.x, rr.y, rr.z)[2], 0.05f) - 0.05f, 2) / 100;

            rays = next;
            funcs.push_back(rays);
            rays.vectorize(x);

        }

        // Intersect the ray with the image sensor.
        Expr X = rays(x, u, l)[0];
        Expr U = rays(x, u, l)[1];
        Expr Z = rays(x, u, l)[2];

        // Walk the ray forward to the right depth.
        X += (sensor_z - Z) * U;
        Z = sensor_z;

        Func next;
        next(x, u, l) = {X, U, Z};

        // Require a minimum height above the sensor
        loss() += pow(min(next(rr.x, rr.y, rr.z)[2] - rays(rr.x, rr.y, rr.z)[2], 0.05f) - 0.05f, 2) / 100;
        rays = next;
        funcs.push_back(rays);
        rays.vectorize(x);

        // Require the rays to come to a focus at the sensor plane.
        RDom rx(0, rays_per_packet, 0, 3);
        Func average_packet_x;
        average_packet_x(u) += rays(rx.x, u, rx.y)[0];
        average_packet_x(u) /= rays_per_packet * 3;

        Func packet_variance;
        packet_variance(u) += pow(rays(rx.x, u, rx.y)[0] - average_packet_x(u), 2);
        packet_variance(u) /= rays_per_packet * 3 - 1; // variance from samples, so -1

        Func spot_size;
        RDom ru(0, packets);
        spot_size() += packet_variance(ru);
        spot_size() /= packets;

        loss() += spot_size();

        // Compute derivatives of the loss with respect to the lens
        // positions and curvatures.
        auto d = Halide::propagate_adjoints(loss);
        Func dr = d(radius);
        Func dz = d(z);

        // This pipeline is tiny. Use a naive schedule.
        funcs.push_back(average_packet_x);
        funcs.push_back(packet_variance);
        funcs.push_back(spot_size);
        for (auto f : funcs) {
            f.compute_root();
        }
        for (auto f : funcs) {
            d(f).compute_root();
        }

        p_optimize = Pipeline({dr, dz, spot_size, loss});
        p_evaluate = Pipeline({spot_size, loss});
    }

    printf("Entering optimization loop\n");
    Buffer<float> dloss_dr(n), dloss_dz(n), old_r(n), old_z(n);
    auto loss_buf = Buffer<float>::make_scalar();
    auto spot_size_buf = Buffer<float>::make_scalar();

    float learning_rate = 0.1f;
    for (int j = 0; j < 1000; j++) {
        // Slowly push the sensor closer to the lens, reoptimizing as we go.
        float sz = std::max(6.0f, 10 - j/100.0f);
        sensor_z.set(sz);

        // Take 50 gradient descent steps in between each move of the sensor.
        for (int i = 0; i < 50; i++) {
            p_optimize.realize({dloss_dr, dloss_dz, spot_size_buf, loss_buf});

            // Do line search in the direction of steepest descent
            int steps = 0;
            for (float l = learning_rate; l > learning_rate / 4096; l /= 2) {
                float prev_loss = loss_buf();
                do {
                    old_r.copy_from(radius_buf);
                    old_z.copy_from(z_buf);
                    for (int s = 0; s < n; s++) {
                        radius_buf(s) -= l * dloss_dr(s);
                        z_buf(s) -= l * dloss_dz(s);
                    }
                    prev_loss = loss_buf();
                    p_evaluate.realize({spot_size_buf, loss_buf});
                    steps++;
                } while (!std::isnan(loss_buf()) && loss_buf() < prev_loss);

                // Undo the last uphill step
                z_buf.copy_from(old_z);
                radius_buf.copy_from(old_r);
                loss_buf() = prev_loss;
            }
            if (steps == 0) {
                learning_rate /= 2;
            }
        }
        printf("Radii: ");
        for (int s = 0; s < n; s++) {
            printf("%f ", radius_buf(s));
        }
        printf("\nDistances: ");
        for (int s = 0; s < n; s++) {
            printf("%f ", z_buf(s));
        }
        printf("\nSpot size: %f\n", sqrt(spot_size_buf()));
        printf("Loss: %g\n", loss_buf());

        if (std::isnan(loss_buf())) break;
    }
}

int main(int argc, char **argv) {
    spherical_lens_system_2d();
    return 0;
}
