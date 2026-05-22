import numpy as np


def orbital_elements_2_cartesian_rv(orbital_elements, gravitational_parameter):

    semimajor, eccentricity, inclination, right_ascension, arg_periapsis, true_anomaly = (
        orbital_elements
    )

    if eccentricity == 0:
        raise ValueError("The current implementation does not admit null eccentricity")

    p = semimajor * (1 - eccentricity**2)
    r = p / (1 + eccentricity * np.cos(true_anomaly))
    r_vect = np.array([r * np.cos(true_anomaly), r * np.sin(true_anomaly), 0.0])
    v_vect = np.sqrt(gravitational_parameter / p) * np.array(
        [-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0]
    )

    cos_Om = np.cos(right_ascension)
    sin_Om = np.sin(right_ascension)
    cos_om = np.cos(arg_periapsis)
    sin_om = np.sin(arg_periapsis)
    cos_i = np.cos(inclination)
    sin_i = np.sin(inclination)

    R = np.array(
        [
            [
                cos_Om * cos_om - sin_Om * sin_om * cos_i,
                -cos_Om * sin_om - sin_Om * cos_om * cos_i,
                sin_Om * sin_i,
            ],
            [
                sin_Om * cos_om + cos_Om * sin_om * cos_i,
                -sin_Om * sin_om + cos_Om * cos_om * cos_i,
                -cos_Om * sin_i,
            ],
            [sin_om * sin_i, cos_om * sin_i, cos_i],
        ]
    )

    ri = R @ r_vect
    vi = R @ v_vect

    return ri, vi


def cartesian_rv_2_orbital_elements(r_vect, v_vect, gravitational_parameter):
    K = np.array([0.0, 0.0, 1.0])

    h_vect = np.cross(r_vect, v_vect)
    n_vect = np.cross(K, h_vect)

    n_norm = np.linalg.norm(n_vect)
    h_norm_sq = np.linalg.norm(h_vect) ** 2
    v_norm_sq = np.linalg.norm(v_vect) ** 2
    r_norm = np.linalg.norm(r_vect)

    e_vect = (1.0 / gravitational_parameter) * (
        (v_norm_sq - gravitational_parameter / r_norm) * r_vect - (r_vect @ v_vect) * v_vect
    )
    p = h_norm_sq / gravitational_parameter
    eccentricity = np.linalg.norm(e_vect)
    e_sq = eccentricity**2
    semimajor = p / (1 - e_sq)

    inclination = np.arccos(h_vect[2] / np.sqrt(h_norm_sq))
    right_ascension = np.arccos(n_vect[0] / n_norm)
    if n_vect[1] < 0 - np.finfo(float).eps:
        right_ascension = 2 * np.pi - right_ascension

    arg_periapsis = np.arccos((n_vect @ e_vect) / (n_norm * eccentricity))
    if e_vect[2] < 0:
        arg_periapsis = 2 * np.pi - arg_periapsis

    true_anomaly = np.arccos((e_vect @ r_vect) / (eccentricity * r_norm))
    if (r_vect @ v_vect) < 0:
        true_anomaly = 2 * np.pi - true_anomaly

    return np.array(
        [semimajor, eccentricity, inclination, right_ascension, arg_periapsis, true_anomaly]
    )


if __name__ == "__main__":
    a = 6800
    e = 0.8
    i = 98 * np.pi / 180
    Om = 15 * np.pi / 180
    om = 50 * np.pi / 180
    theta = 90 * np.pi / 180
    gravitational_parameter = 3.98e5

    orbital_elements = (a, e, i, Om, om, theta)
    r_vect, v_vect = orbital_elements_2_cartesian_rv(orbital_elements, gravitational_parameter)

    print(f"r_norm   = {np.linalg.norm(r_vect):.3f}")
    print(f"r_vect_x = {r_vect[0]:.3f}")
    print(f"r_vect_y = {r_vect[1]:.3f}")
    print(f"r_vect_z = {r_vect[2]:.3f}")

    print(f"v_norm      = {np.linalg.norm(v_vect):.3f}")
    print(f"v_vect_x    = {v_vect[0]:.3f}")
    print(f"v_vect_y    = {v_vect[1]:.3f}")
    print(f"v_vect_z    = {v_vect[2]:.3f}")

    orbital_elements_out = cartesian_rv_2_orbital_elements(r_vect, v_vect, gravitational_parameter)

    list_names = [
        "semimajor",
        "eccentricity",
        "inclination",
        "right_ascension",
        "arg_periapsis",
        "true_anomaly",
    ]
    for i in range(len(list_names)):
        print(
            [
                list_names[i],
                f"orbital_in:{orbital_elements[i]:.3f}",
                f"orbital_out:{orbital_elements_out[i]:.3f}",
            ]
        )

    for i in range(len(orbital_elements_out)):
        assert np.linalg.norm(orbital_elements_out[i] - orbital_elements[i]) <= np.sqrt(
            np.finfo("float").eps
        )
