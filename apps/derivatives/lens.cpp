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
    float eps = 1e-20f;
    d = select(d <= eps, 0, sqrt(max(d, eps)));
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

void spherical_lens_system_2d() {
    // Multiple spherical surfaces which alternate between air ->
    // glass and glass -> air.

    int n; // number of surfaces
    double fov; // The maximum ray slope at the aperture. A measure of field of view.
    double F; // Proportional to the extent of the ray packet at the aperture. The F-number.
    Buffer<double> radius_buf; // radius of each surface
    Buffer<double> z_buf;      // distance between element i and element i+1 (element n is the image sensor).
    Buffer<double> N_buf;      // Relative index of refraction of each surface, at 3 different wavelengths.
    Buffer<double> z_min;      // Minimum permissible value for z_buf. We don't want infinitely thin lenses.
    Param<double> sensor_z;    // The depth of the image sensor

    // Initialize the system
    if (0) { // Double-Gauss
        // Some useful constants for indices of refraction of common lens
        // materials. From http://refractiveindex.info
        /*
          const double bk7_380 = 1.5337f;
          const double bk7_550 = 1.5185f;
          const double bk7_750 = 1.5118f;
          const double f2_380 = 1.6595f;
          const double f2_550 = 1.6237f;
          const double f2_750 = 1.6103f;
        */

        // The indices of refraction have to match for an achromatic
        // doublet, so the above can't be right. Values below taken from
        // the Zeiss patent reproduced here
        // http://www.reduser.net/forum/showthread.php?154412-Who-is-the-father-of-all-fast-50mm-lenses-Planar-vs-Opic-Lens-evolution

        const double flint_380 = 1.59227f;
        const double flint_750 = 1.57631f;
        const double flint_550 = (flint_380 + flint_750)/2;
        const double crown_380 = 1.58512f;
        const double crown_750 = 1.57244f;
        const double crown_550 = (crown_380 + crown_750)/2;

        n = 10;
        radius_buf = Buffer<double>(n);
        z_buf = Buffer<double>(n);
        z_min = Buffer<double>(n);
        N_buf = Buffer<double>(n, 3);

        double scale = 8.6;

        // Set up the initial layer thicknesses
        z_buf(0) = 0.f;
        z_buf(1) = 0.0308 * scale;
        z_buf(2) = 0.0026 * scale;
        z_buf(3) = 0.0514 * scale;
        z_buf(4) = 0.0411 * scale;
        z_buf(5) = 2 * 0.0514 * scale;
        z_buf(6) = 0.0411 * scale;
        z_buf(7) = 0.0514 * scale;
        z_buf(8) = 0.0026 * scale;
        z_buf(9) = 0.0308 * scale;

        z_min(0) = 0.0f; // air before first element
        z_min(1) = 0.1f; // glass
        z_min(2) = 0.0f; // air
        z_min(3) = 0.1f; // glass
        z_min(4) = 0.1f; // glass
        z_min(5) = 0.1f; // air at the aperture
        z_min(6) = 0.1f; // glass
        z_min(7) = 0.1f; // glass
        z_min(8) = 0.0f; // air
        z_min(9) = 0.1f; // glass

        // Set the initial radii, according to the patent
        radius_buf(0) = 0.3147f * scale;
        radius_buf(1) = 1.5424f * scale;
        radius_buf(2) = 0.3599f * scale;
        radius_buf(3) = -0.4370f * scale;
        radius_buf(4) = 0.1954f * scale;
        radius_buf(5) = -0.1954f * scale;
        radius_buf(6) = 0.4370f * scale;
        radius_buf(7) = -0.3599f * scale;
        radius_buf(8) = -1.5424f * scale;
        radius_buf(9) = -0.3147f * scale;

        // Set the relative indices of refraction for each surface. A
        // single Gauss system goes FCAC, where C = crown glass, A =
        // air, F = flint glass
        N_buf(0, 0) = crown_380;        // A -> C
        N_buf(0, 1) = crown_550;
        N_buf(0, 2) = crown_750;
        N_buf(1, 0) = 1/crown_380;      // C -> A
        N_buf(1, 1) = 1/crown_550;
        N_buf(1, 2) = 1/crown_750;
        N_buf(2, 0) = crown_380;        // A -> C
        N_buf(2, 1) = crown_550;
        N_buf(2, 2) = crown_750;
        N_buf(3, 0) = flint_380/crown_380; // C -> F
        N_buf(3, 1) = flint_550/crown_550;
        N_buf(3, 2) = flint_750/crown_750;
        N_buf(4, 0) = 1/flint_380;         // F -> A
        N_buf(4, 1) = 1/flint_550;
        N_buf(4, 2) = 1/flint_750;
        N_buf(5, 0) = flint_380;         // A -> F
        N_buf(5, 1) = flint_550;
        N_buf(5, 2) = flint_750;
        N_buf(6, 0) = crown_380/flint_380; // F -> C
        N_buf(6, 1) = crown_550/flint_550;
        N_buf(6, 2) = crown_750/flint_750;
        N_buf(7, 0) = 1/crown_380;      // C -> A
        N_buf(7, 1) = 1/crown_550;
        N_buf(7, 2) = 1/crown_750;
        N_buf(8, 0) = crown_380;        // A -> C
        N_buf(8, 1) = crown_550;
        N_buf(8, 2) = crown_750;
        N_buf(9, 0) = 1/crown_380;      // C -> A
        N_buf(9, 1) = 1/crown_550;
        N_buf(9, 2) = 1/crown_750;
        fov = 1/8.0;
        F = 8.0;
    }

    {
        // Zeiss 1955 design from US2799207A
        // Constants from the patent:
        double
            r1 = 0.347219, r2 = 0.904131, r3 = 0.309462, r4 = 0.215955,
            r5 = -0.228743, r6 = 1.69089, r7 = -0.320794, r8 = -2.83857,
            r9 = -0.538549, d1 = 0.07573, l1 = 0.00132, d2 = 0.06236,
            l2 = 0.24492, d3 = 0.01324, d4 = 0.09492, l3 = 0.00132,
            d5 = 0.06620, n1 = 1.62041, v1 = 60.3, n2 = 1.75520,
            v2 = 27.5, n3 = 1.71736, v3 = 29.5, n4 = 1.69067,
            v4 = 54.9, n5 = 1.75520, v5 = 27.5;

        n = 9;
        radius_buf = Buffer<double>(n);
        z_buf = Buffer<double>(n);
        z_min = Buffer<double>(n);
        N_buf = Buffer<double>(n, 3);

        // Scale up the focal length to 10.
        double scale = 7.715395;

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

        z_min.copy_from(z_buf);
        for (int i = 0; i < 9; i++) {
            z_min(i) /= 2;
        }

        // Integrate to get absolute coordinates of each element's intersection with the optical axis
        for (int i = 1; i < 9; i++) {
            z_buf(i) += z_buf(i-1);
        }

        // If we assume that for the material, nD = (nF + nC)/c, the
        // Abbe number and nD are sufficient to calculate an index of
        // refraction at all three wavelengths.
        double dn1 = (n1 - 1) / (2 * v1);
        double dn2 = (n2 - 1) / (2 * v2);
        double dn3 = (n3 - 1) / (2 * v3);
        double dn4 = (n4 - 1) / (2 * v4);
        double dn5 = (n5 - 1) / (2 * v5);

        // Absolute indices of refraction
        N_buf.fill(1.0); // Initialize to air
        N_buf(0, 0) = n1 - dn1; // A -> C
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

        // Convert to relative
        for (int i = n-1; i > 0; i--) {
            for (int j = 0; j < 3; j++) {
                N_buf(i, j) /= N_buf(i-1, j);
            }
        }

        fov = 1/6.0;
        F = 6.0;
    }


    // We'll compile three pipeline variants - one for doing the
    // optimization, one just for evaluating loss, and one for
    // generating postscript figures.

    Pipeline p_optimize, p_evaluate, p_render;

    for (int gen_postscript = 0; gen_postscript < 2; gen_postscript++) {

        const int rays_per_packet = gen_postscript ? 4 : 8;
        const int packets = gen_postscript ? 4 : 8;

        Var x, u, l;

        Func radius;
        radius(x) = radius_buf(x);

        Func z;
        z(x) = z_buf(x);

        Func N;
        N(x, l) = N_buf(x, l);

        // Maximum ray slope of fov
        Func initial_u;
        initial_u(u) = ((cast<double>(2*u))/(packets - 1) - 1) * Expr(fov);

        double focal_length = 10; // distance from aperture to sensor
        double aperture_diameter = focal_length / F;
        double aperture_radius = aperture_diameter / 2;
        Func initial_x;
        initial_x(x) = (cast<double>(2*x)/(rays_per_packet - 1) - 1) * Expr(aperture_radius);

        Func rays;
        rays(x, u, l) = {initial_x(x), initial_u(u), Expr(0.0)}; // rays(x, u, l) for fixed u is a 'packet' of parallel rays at wavelength l

        vector<Func> funcs;
        rays.vectorize(x);
        funcs.push_back(rays);

        Func loss;
        loss() = Expr(0.0);

        RDom rr(0, rays_per_packet, 0, packets, 0, 3);

        for (int i = 0; i < n; i++) {
            // Intersect with the next lens surface
            Expr X = rays(x, u, l)[0], U = rays(x, u, l)[1], Z = rays(x, u, l)[2];
            Expr R = radius(i);

            Expr X_initial = X;
            Expr Z_initial = Z;

            // Walk the ray the reference plane for this surface
            X += (z(i) - Z) * U;
            Z = z(i);

            // Walk the ray forward until it intersects the surface
            auto roots = solve_quadratic(U*U + 1, 2*(X*U - R), X*X);
            Expr alpha = select(R < 0, roots.first, roots.second);
            X += alpha * U;
            Z += alpha;

            // Compute the slope of the normal to the lens surface at the intersection
            Expr u3 = -X / (R - alpha);

            // Compute the slope of the outgoing ray
            Expr u1 = U;
            Expr A = N(i, l)*N(i, l) * (1 + u1*u1);
            Expr B = pow(u3 - u1, 2);
            roots = solve_quadratic(A - B, 0 - 2*A*u3, A*u3*u3 - B);
            Expr u2 = select((u1 < u3) ^ (roots.first < u3), roots.second, roots.first);

            // Refract!
            U = u2;

            if (gen_postscript) {

                // If this is the first surface, start the rays a little early to make the figure clearer
                if (i == 0) {
                    Z_initial -= 1;
                    X_initial -= rays(x, u, l)[1];
                }

                // Emit the rays as postscript as a side-effect (Whee hack!)
                X = print(X, "pop newpath",
                          select(l == 0, 0.8f, 0.0f),
                          select(l == 1, 0.8f, 0.0f),
                          select(l == 2, 0.8f, 0.0f), "setrgbcolor",
                          Z_initial, X_initial, "moveto",
                          Z, X, "lineto stroke",
                          "% x =", x, "u =", u, "l =", l);
            }

            Func next;
            next(x, u, l) = {X, U, Z};

            // No reflections, please
            loss() += pow(min(next(rr.x, rr.y, rr.z)[2] - rays(rr.x, rr.y, rr.z)[2], Expr(0.05)) - Expr(0.05), 2) / 100;

            rays = next;
            funcs.push_back(rays);
            rays.vectorize(x);

            if (i == n-1) {
                X = rays(x, u, l)[0];
                U = rays(x, u, l)[1];
                Z = rays(x, u, l)[2];

                // Walk the ray forward to the image sensor
                X += (sensor_z - Z) * U;
                Z = sensor_z;

                if (gen_postscript) {
                    X = print(X, "pop newpath",
                              select(l == 0, 0.8f, 0.0f),
                              select(l == 1, 0.8f, 0.0f),
                              select(l == 2, 0.8f, 0.0f), "setrgbcolor",
                              rays(x, u, l)[2], rays(x, u, l)[0], "moveto",
                              Z, X, "lineto stroke",
                              "% x =", x, "u =", u, "l =", l);
                }

                Func next;
                next(x, u, l) = {X, U, Z};
                // No reflections, please
                loss() += pow(min(next(rr.x, rr.y, rr.z)[2] - rays(rr.x, rr.y, rr.z)[2], Expr(0.05)) - Expr(0.05), 2) / 100;
                rays = next;
                funcs.push_back(rays);
                rays.vectorize(x);
            }
        }

        Func sensor = rays;

        // Require the rays to come to a focus.
        RDom rx(0, rays_per_packet, 0, 3);
        Func average_packet_x;
        average_packet_x(u) += sensor(rx.x, u, rx.y)[0];
        average_packet_x(u) /= rays_per_packet * 3;

        Func packet_variance;
        packet_variance(u) += pow(sensor(rx.x, u, rx.y)[0] - average_packet_x(u), 2);
        packet_variance(u) /= rays_per_packet * 3 - 1; // variance from samples, so -1

        Func spot_size;
        RDom ru(0, packets);
        spot_size() += packet_variance(ru);
        spot_size() /= packets;

        loss() += spot_size();

        /*
        // Constrain the gaps to be positive
        for (int i = 0; i < n - 1; i++) {
            Expr prev = (i > 0) ? z(i-1) : Expr(0.0);
            loss() += pow(min((z(i) - prev) - Expr(z_min(i)), 0), 2);
        }
        */

        funcs.push_back(average_packet_x);
        funcs.push_back(packet_variance);
        funcs.push_back(spot_size);
        //funcs.push_back(loss);

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
            p_render = Pipeline({spot_size, loss});
            p_render.set_custom_print(print_to_stream);
        } else {
            auto d = Halide::propagate_adjoints(loss);

            for (auto f : funcs) {
                d(f).compute_root();
            }

            Func dr = d(radius);
            Func dz = d(z);

            p_optimize = Pipeline({dr, dz, spot_size, loss});
            p_evaluate = Pipeline({spot_size, loss});
        }
    }

    fprintf(stderr, "Entering optimization loop\n");
    Buffer<double> dloss_dr(n), dloss_dz(n);
    auto loss_buf = Buffer<double>::make_scalar();
    auto spot_size_buf = Buffer<double>::make_scalar();
    Buffer<double> old_r(n), old_z(n);

    double learning_rate = 0.1;
    for (int j = 0; j < 1000; j++) {
        double sz = std::max(6.0, 10 - j/100.0);
        sensor_z.set(sz);

        std::ofstream postscript("iter_" + std::to_string(j) + ".ps", std::ios_base::trunc);
        stream = &postscript;
        postscript << "<</PageSize [1200 675] >> setpagedevice\n"
                   << "100 337 translate\n"
                   << "100 100 scale\n"
                   << "0.002 setlinewidth\n";
        // Draw the lens surfaces
        for (int s = 0; s < n; s++) {
            double r = radius_buf(s);
            double theta = 180 * (asin(1.4 / std::max(1.41, std::abs(r))) / M_PI);
            postscript << "newpath ";
            if (r > 0) {
                postscript << (z_buf(s) + r) << " 0 " << r << " " << (180 - theta) << " " << (180 + theta);
            } else {
                postscript << (z_buf(s) + r) << " 0 " << std::abs(r) << " " << -theta << " " << theta;
            }
            postscript << " arc stroke\n";
        }
        // Draw the sensor
        postscript << "newpath " << sz << " 4 moveto " << sz << " -4 lineto stroke\n";

        double old_loss = loss_buf();

        p_render.realize({spot_size_buf, loss_buf});
        postscript << "stroke showpage\n";
        stream = nullptr;
        postscript.close();

        for (int i = 0; i < 20; i++) {
            p_optimize.realize({dloss_dr, dloss_dz, spot_size_buf, loss_buf});

            // Do line search in this direction
            int steps = 0;
            for (double l = learning_rate; l > learning_rate / 4096; l /= 2) {
                double prev_loss = loss_buf();
                while (!std::isnan(loss_buf()) && loss_buf() <= prev_loss) {
                    old_r.copy_from(radius_buf);
                    old_z.copy_from(z_buf);
                    for (int s = 0; s < n; s++) {
                        radius_buf(s) -= l * dloss_dr(s);
                        z_buf(s) -= l * dloss_dz(s);
                    }
                    prev_loss = loss_buf();
                    p_evaluate.realize({spot_size_buf, loss_buf});
                    steps++;
                }
                // Undo the last bad step
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

    //thin_lens();

    //spherical_surface_2d();

    spherical_lens_system_2d();

    return 0;
}
