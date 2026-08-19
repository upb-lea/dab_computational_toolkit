# -*- coding: utf-8 -*-
"""
Generate a round PQ40/40-style bobbin and export it as STEP using FreeCAD.

The bobbin consists of:
- round center-leg hole
- cylindrical winding barrel
- circular top and bottom flanges
- rounded inner hole edges
- rounded outer top and bottom flange edges
- optional wire exit slots on the positive Y side

All dimensions are in mm.

Environment variables
---------------------
WINDOW_H_MM
WINDOW_W_MM
CORE_INNER_DIAMETER_MM
FLANGE_THICKNESS_MM
CLEARANCE
INNER_EDGE_RADIUS
OUTER_EDGE_RADIUS
ENABLE_WIRE_SLOTS
WIRE_SLOT_WIDTH
WIRE_SLOT_DEPTH
WIRE_SLOT_HEIGHT
WIRE_SLOTS_POSITION
OUTPUT_STEP_FILE
SAVE_FCSTD_FILE

WIRE_SLOTS_POSITION may be:
- "both"
- "top"
- "bottom"
- "none"

Example:
WINDOW_H_MM=17.2 \
WINDOW_W_MM=7.0 \
CORE_INNER_DIAMETER_MM=13.45 \
OUTPUT_STEP_FILE="./pq4040_bobbin.step" \
FreeCADCmd pq4040_round_bobbin.py
"""

# Python libraries
import os
import sys
import logging

# FreeCAD libraries
import FreeCAD as App
import Part
import Import


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_float_environment_variable(variable_name: str, default_value: float) -> float:
    """
    Read a floating-point value from an environment variable.

    The default is used when the variable is absent or empty.

    :param variable_name: variable name
    :type variable_name: str
    :param default_value: default value
    :type default_value: float
    """
    value = os.environ.get(variable_name)

    if value is None or value.strip() == "":
        return float(default_value)

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"Environment variable {variable_name} must be numeric, "
            f"but received {value!r}."
        ) from error


def read_bool_environment_variable(variable_name: str, default_value: bool) -> bool:
    """
    Read a boolean value from an environment variable.

    :param variable_name: variable name
    :type variable_name: str
    :param default_value: default value
    :type default_value: bool

    Accepted true values:
        1, true, yes, on

    Accepted false values:
        0, false, no, off
    """
    value = os.environ.get(variable_name)

    if value is None or value.strip() == "":
        return bool(default_value)

    value = value.strip().lower()

    if value in ("1", "true", "yes", "on"):
        return True

    if value in ("0", "false", "no", "off"):
        return False

    raise ValueError(
        f"Environment variable {variable_name} must be boolean, "
        f"but received {value!r}."
    )


def read_string_environment_variable(variable_name: str, default_value: str) -> str:
    """
    Read a string environment variable.

    :param variable_name: variable name
    :type variable_name: str
    :param default_value: default value
    :type default_value: str
    """
    value = os.environ.get(variable_name)

    if value is None or value.strip() == "":
        return default_value

    return value.strip()


def vector_from_rz(radius_mm: float, z_mm: float) -> App.Vector:
    """
    Create a point in the radial X/Z profile plane.

    The X coordinate represents the radial distance from the global Z-axis.
    The Z coordinate represents the axial position. The Y coordinate is zero.
    The resulting point is intended for a profile revolved around the global
    Z-axis.

    :param radius_mm: Radial distance from the global Z-axis in mm.
    :type radius_mm: float
    :param z_mm: Axial Z coordinate in mm.
    :type z_mm: float
    :return: FreeCAD vector in the X/Z profile plane.
    :rtype: App.Vector
    """
    return App.Vector(radius_mm, 0.0, z_mm)


def make_arc_edge(
    start_point: App.Vector,
    middle_point: App.Vector,
    end_point: App.Vector
) -> Part.TopoShape:
    """
    Create a circular arc edge through three points.

    The arc starts at ``start_point``, passes through ``middle_point``,
    and ends at ``end_point``.

    :param start_point: Start point of the circular arc.
    :type start_point: App.Vector
    :param middle_point: Point on the circular arc between start and end.
    :type middle_point: App.Vector
    :param end_point: End point of the circular arc.
    :type end_point: App.Vector
    :return: FreeCAD shape representing the circular arc edge.
    :rtype: Part.TopoShape
    """
    return Part.Arc(
        start_point,
        middle_point,
        end_point
    ).toShape()


# ---------------------------------------------------------------------------
# Geometry creation
# ---------------------------------------------------------------------------

def create_round_bobbin(
    window_h_mm: float,
    window_w_mm: float,
    core_inner_diameter_mm: float,
    flange_thickness_mm: float,
    clearance: float,
    inner_edge_radius_mm: float,
    outer_edge_radius_mm: float,
    enable_wire_slots: bool,
    wire_slot_width_mm: float,
    wire_slot_depth_mm: float,
    wire_slot_height_mm: float,
    wire_slots_position: str
) -> Part.TopoShape:
    """
    Create the complete round bobbin as a Part.Shape.

    The bobbin profile is built in the X/Z plane and then revolved around
    the global Z axis.

    :param window_h_mm: available core window height
    :param window_w_mm: radial core window width
    :param core_inner_diameter_mm: center-leg diameter
    :param flange_thickness_mm: top and bottom flange thickness
    :param clearance: bobbin clearance to the core
    :param inner_edge_radius_mm: rounding radius at center-hole edges
    :param outer_edge_radius_mm: rounding radius at outer flange edges
    :param enable_wire_slots: whether wire slots are cut
    :param wire_slot_width_mm: tangential slot width
    :param wire_slot_depth_mm: slot depth from the flange exterior
    :param wire_slot_height_mm: axial slot height
    :param wire_slots_position: "both", "top", "bottom", or "none"
    :return: final bobbin solid
    """
    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------
    if window_h_mm <= 0:
        raise ValueError("window_h_mm must be greater than 0.")

    if window_w_mm <= 0:
        raise ValueError("window_w_mm must be greater than 0.")

    if core_inner_diameter_mm <= 0:
        raise ValueError(
            "core_inner_diameter_mm must be greater than 0."
        )

    if flange_thickness_mm <= 0:
        raise ValueError(
            "flange_thickness_mm must be greater than 0."
        )

    if clearance < 0:
        raise ValueError("clearance must not be negative.")

    if inner_edge_radius_mm < 0:
        raise ValueError(
            "inner_edge_radius_mm must not be negative."
        )

    if outer_edge_radius_mm < 0:
        raise ValueError(
            "outer_edge_radius_mm must not be negative."
        )

    wire_slots_position = wire_slots_position.strip().lower()

    if wire_slots_position not in ("both", "top", "bottom", "none"):
        raise ValueError(
            "wire_slots_position must be one of: "
            "'both', 'top', 'bottom', 'none'."
        )

    # -----------------------------------------------------------------------
    # Derived dimensions
    # -----------------------------------------------------------------------
    bobbin_height_mm = window_h_mm - 2 * clearance

    winding_height_mm = bobbin_height_mm - 2 * flange_thickness_mm

    center_hole_diameter_mm = core_inner_diameter_mm + 2 * clearance

    barrel_outer_diameter_mm = core_inner_diameter_mm + 2 * clearance + 2 * flange_thickness_mm

    flange_outer_diameter_mm = core_inner_diameter_mm + 2 * window_w_mm - 2 * clearance

    center_hole_radius_mm = center_hole_diameter_mm / 2
    barrel_outer_radius_mm = barrel_outer_diameter_mm / 2
    flange_outer_radius_mm = flange_outer_diameter_mm / 2

    z_min_mm = -bobbin_height_mm / 2
    z_max_mm = bobbin_height_mm / 2

    # -----------------------------------------------------------------------
    # Derived geometry validation
    # -----------------------------------------------------------------------
    if bobbin_height_mm <= 0:
        raise ValueError(
            "Invalid bobbin height. WINDOW_H_MM must be greater than "
            "2 * CLEARANCE."
        )

    if winding_height_mm <= 0:
        raise ValueError(
            "Invalid winding height. Reduce FLANGE_THICKNESS_MM or "
            "increase WINDOW_H_MM."
        )

    if barrel_outer_radius_mm <= center_hole_radius_mm:
        raise ValueError(
            "Invalid geometry: barrel outer radius must be larger than "
            "the center-hole radius."
        )

    if flange_outer_radius_mm <= barrel_outer_radius_mm:
        raise ValueError(
            "Invalid geometry: flange outer radius must be larger than "
            "the barrel outer radius."
        )

    # Limit edge radii
    inner_roundover_radius_mm = min(
        inner_edge_radius_mm,
        flange_thickness_mm * 0.95,
        winding_height_mm / 2.0,
        barrel_outer_radius_mm - center_hole_radius_mm
    )

    outer_roundover_radius_mm = min(
        outer_edge_radius_mm,
        flange_thickness_mm * 0.95,
        flange_outer_radius_mm - barrel_outer_radius_mm
    )

    if inner_roundover_radius_mm <= 0:
        raise ValueError(
            "Computed inner roundover radius is invalid."
        )

    if outer_roundover_radius_mm <= 0:
        raise ValueError(
            "Computed outer roundover radius is invalid."
        )

    # -----------------------------------------------------------------------
    # Build radial profile in the X/Z plane
    # -----------------------------------------------------------------------
    # Bottom face: center-hole-side to outer-flange-side.
    p0 = vector_from_rz(
        center_hole_radius_mm + inner_roundover_radius_mm,
        z_min_mm
    )
    p1 = vector_from_rz(
        flange_outer_radius_mm - outer_roundover_radius_mm,
        z_min_mm
    )

    # Bottom outer roundover: -90 degrees to 0 degrees.
    p2 = vector_from_rz(
        flange_outer_radius_mm,
        z_min_mm + outer_roundover_radius_mm
    )
    p_bottom_outer_mid = vector_from_rz(
        flange_outer_radius_mm - outer_roundover_radius_mm + outer_roundover_radius_mm * 0.7071067811865476,
        z_min_mm + outer_roundover_radius_mm - outer_roundover_radius_mm * 0.7071067811865476
    )

    # Lower outer flange and transition to winding barrel.
    p3 = vector_from_rz(
        flange_outer_radius_mm,
        z_min_mm + flange_thickness_mm
    )
    p4 = vector_from_rz(
        barrel_outer_radius_mm,
        z_min_mm + flange_thickness_mm
    )

    # Winding barrel.
    p5 = vector_from_rz(
        barrel_outer_radius_mm,
        z_max_mm - flange_thickness_mm
    )

    # Upper flange.
    p6 = vector_from_rz(
        flange_outer_radius_mm,
        z_max_mm - flange_thickness_mm
    )
    p7 = vector_from_rz(
        flange_outer_radius_mm,
        z_max_mm - outer_roundover_radius_mm
    )

    # Top outer roundover: 0 degrees to 90 degrees.
    p8 = vector_from_rz(
        flange_outer_radius_mm - outer_roundover_radius_mm,
        z_max_mm
    )
    p_top_outer_mid = vector_from_rz(
        flange_outer_radius_mm - outer_roundover_radius_mm + outer_roundover_radius_mm * 0.7071067811865476,
        z_max_mm - outer_roundover_radius_mm + outer_roundover_radius_mm * 0.7071067811865476
    )

    # Top face: outer flange to center-hole-side.
    p9 = vector_from_rz(
        center_hole_radius_mm + inner_roundover_radius_mm,
        z_max_mm
    )

    # Top inner roundover: 90 degrees to 180 degrees.
    p10 = vector_from_rz(
        center_hole_radius_mm,
        z_max_mm - inner_roundover_radius_mm
    )
    p_top_inner_mid = vector_from_rz(
        center_hole_radius_mm + inner_roundover_radius_mm - inner_roundover_radius_mm * 0.7071067811865476,
        z_max_mm - inner_roundover_radius_mm + inner_roundover_radius_mm * 0.7071067811865476
    )

    # Inner center-hole wall.
    p11 = vector_from_rz(
        center_hole_radius_mm,
        z_min_mm + inner_roundover_radius_mm
    )

    # Bottom inner roundover: 180 degrees to 270 degrees.
    p_bottom_inner_mid = vector_from_rz(
        center_hole_radius_mm + inner_roundover_radius_mm - inner_roundover_radius_mm * 0.7071067811865476,
        z_min_mm + inner_roundover_radius_mm - inner_roundover_radius_mm * 0.7071067811865476
    )

    # Create the closed profile wire.
    profile_edges = [
        Part.makeLine(p0, p1),

        make_arc_edge(
            p1,
            p_bottom_outer_mid,
            p2
        ),

        Part.makeLine(p2, p3),
        Part.makeLine(p3, p4),
        Part.makeLine(p4, p5),
        Part.makeLine(p5, p6),
        Part.makeLine(p6, p7),

        make_arc_edge(
            p7,
            p_top_outer_mid,
            p8
        ),

        Part.makeLine(p8, p9),

        make_arc_edge(
            p9,
            p_top_inner_mid,
            p10
        ),

        Part.makeLine(p10, p11),

        make_arc_edge(
            p11,
            p_bottom_inner_mid,
            p0
        )
    ]

    profile_wire = Part.Wire(profile_edges)

    if not profile_wire.isClosed():
        raise RuntimeError("Internal error: bobbin profile wire is not closed.")

    profile_face = Part.Face(profile_wire)

    # Revolve the radial profile around the Z axis.
    bobbin_shape = profile_face.revolve(
        App.Vector(0.0, 0.0, 0.0),
        App.Vector(0.0, 0.0, 1.0),
        360.0
    )

    # -----------------------------------------------------------------------
    # Optional wire exit slots
    # -----------------------------------------------------------------------
    if enable_wire_slots and wire_slots_position != "none":
        if wire_slot_width_mm <= 0:
            raise ValueError(
                "wire_slot_width_mm must be greater than 0."
            )

        if wire_slot_depth_mm <= 0:
            raise ValueError(
                "wire_slot_depth_mm must be greater than 0."
            )

        if wire_slot_height_mm <= 0:
            raise ValueError(
                "wire_slot_height_mm must be greater than 0."
            )


        # y = flange_outer_radius - wire_slot_depth / 2
        # and has a Y length of wire_slot_depth + 0.4.
        # Therefore, it begins 0.2 mm outside the flange outer radius,
        # guaranteeing a clean Boolean cut.
        slot_x_min_mm = -wire_slot_width_mm / 2.0
        slot_y_min_mm = (flange_outer_radius_mm - wire_slot_depth_mm - 0.2)
        slot_y_length_mm = wire_slot_depth_mm + 0.4

        def make_wire_slot(slot_z_center_mm: float) -> Part.TopoShape:
            """
            Create a rectangular cutting solid for one wire exit slot.

            The slot is positioned on the positive Y side of the bobbin flange
            and extends slightly beyond the outer flange edge to ensure a
            clean Boolean cut.

            :param slot_z_center_mm: Axial Z coordinate of the slot center
                in mm.
            :type slot_z_center_mm: float
            :return: Rectangular FreeCAD solid used to cut the wire slot.
            :rtype: Part.TopoShape
            """
            slot_z_min_mm = (
                slot_z_center_mm - wire_slot_height_mm / 2.0
            )

            return Part.makeBox(
                wire_slot_width_mm,
                slot_y_length_mm,
                wire_slot_height_mm,
                App.Vector(
                    slot_x_min_mm,
                    slot_y_min_mm,
                    slot_z_min_mm
                )
            )

        slot_cutters = []

        if wire_slots_position in ("bottom", "both"):
            bottom_slot_z_center_mm = (
                z_min_mm + flange_thickness_mm / 2.0
            )
            slot_cutters.append(
                make_wire_slot(bottom_slot_z_center_mm)
            )

        if wire_slots_position in ("top", "both"):
            top_slot_z_center_mm = (
                z_max_mm - flange_thickness_mm / 2.0
            )
            slot_cutters.append(
                make_wire_slot(top_slot_z_center_mm)
            )

        for slot_cutter in slot_cutters:
            bobbin_shape = bobbin_shape.cut(slot_cutter)

    # Remove unnecessary Boolean splitter edges.
    bobbin_shape = bobbin_shape.removeSplitter()

    if bobbin_shape.isNull():
        raise RuntimeError("Generated bobbin shape is empty.")

    if not bobbin_shape.isValid():
        raise RuntimeError("Generated bobbin shape is invalid.")

    return bobbin_shape


def export_round_bobbin_step(
    window_h_mm: float,
    window_w_mm: float,
    core_inner_diameter_mm: float,
    flange_thickness_mm: float,
    clearance: float,
    inner_edge_radius_mm: float,
    outer_edge_radius_mm: float,
    enable_wire_slots: bool,
    wire_slot_width_mm: float,
    wire_slot_depth_mm: float,
    wire_slot_height_mm: float,
    wire_slots_position: str,
    output_step_file: str,
    save_freecad_file: bool = False
) -> str:
    """
    Create a round PQ40/40-style bobbin and export it as a STEP file.

    The bobbin includes a round center-leg hole, a cylindrical winding barrel,
    circular upper and lower flanges, rounded inner and outer flange edges,
    and optionally one or two wire exit slots.

    :param window_h_mm: Available height of the magnetic-core window in mm.
    :type window_h_mm: float
    :param window_w_mm: Radial width of the magnetic-core window in mm.
    :type window_w_mm: float
    :param core_inner_diameter_mm: Diameter of the magnetic core center leg
        in mm.
    :type core_inner_diameter_mm: float
    :param flange_thickness_mm: Thickness of both the top and bottom bobbin
        flanges in mm.
    :type flange_thickness_mm: float
    :param clearance: Required clearance between bobbin and magnetic core in
        mm.
    :type clearance: float
    :param inner_edge_radius_mm: Radius of the rounded center-hole edges in
        mm.
    :type inner_edge_radius_mm: float
    :param outer_edge_radius_mm: Radius of the rounded outer top and bottom
        flange edges in mm.
    :type outer_edge_radius_mm: float
    :param enable_wire_slots: Whether wire exit slots are cut into the
        flanges.
    :type enable_wire_slots: bool
    :param wire_slot_width_mm: Tangential width of each wire exit slot in mm.
    :type wire_slot_width_mm: float
    :param wire_slot_depth_mm: Radial depth of each wire exit slot, measured
        from the outer flange edge towards the bobbin center, in mm.
    :type wire_slot_depth_mm: float
    :param wire_slot_height_mm: Axial height of each wire exit slot in mm.
    :type wire_slot_height_mm: float
    :param wire_slots_position: Position of wire exit slots. Allowed values
        are ``"both"``, ``"top"``, ``"bottom"``, and ``"none"``.
    :type wire_slots_position: str
    :param output_step_file: Destination path of the generated STEP file.
    :type output_step_file: str
    :param save_freecad_file: True to additionally save an editable FreeCAD
        FCStd document next to the STEP file.
    :type save_freecad_file: bool
    :return: Absolute path of the generated STEP file.
    :rtype: str
    :raises ValueError: If the output path is empty or parameters result in
        invalid geometry.
    :raises RuntimeError: If bobbin creation fails or the STEP file is not
        created successfully.
    """
    if not output_step_file:
        raise ValueError("output_step_file must not be empty.")

    output_step_file = os.path.abspath(output_step_file)
    output_directory = os.path.dirname(output_step_file)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    document = App.newDocument("PQ4040_Round_Bobbin")

    try:
        final_shape = create_round_bobbin(
            window_h_mm=window_h_mm,
            window_w_mm=window_w_mm,
            core_inner_diameter_mm=core_inner_diameter_mm,
            flange_thickness_mm=flange_thickness_mm,
            clearance=clearance,
            inner_edge_radius_mm=inner_edge_radius_mm,
            outer_edge_radius_mm=outer_edge_radius_mm,
            enable_wire_slots=enable_wire_slots,
            wire_slot_width_mm=wire_slot_width_mm,
            wire_slot_depth_mm=wire_slot_depth_mm,
            wire_slot_height_mm=wire_slot_height_mm,
            wire_slots_position=wire_slots_position
        )

        # This is intentionally a Part::Feature, matching the working
        # PQ-core script and providing robust STEP export behavior.
        bobbin_object = document.addObject(
            "Part::Feature",
            "PQ_Round_Bobbin"
        )
        bobbin_object.Label = "PQ Round Bobbin"
        bobbin_object.Shape = final_shape

        document.recompute()

        # Export the document object, consistent with the core generator.
        Import.export([bobbin_object], output_step_file)

        if not os.path.isfile(output_step_file):
            raise RuntimeError(f"FreeCAD did not create the STEP file: {output_step_file}")

        if os.path.getsize(output_step_file) == 0:
            raise RuntimeError(f"FreeCAD created an empty STEP file: {output_step_file}")

        if save_freecad_file:
            output_freecad_file = (
                os.path.splitext(output_step_file)[0] + ".FCStd"
            )
            document.saveAs(output_freecad_file)
            logger.info(f"FreeCAD document created: {output_freecad_file}")

        logger.info(f"STEP file created: {output_step_file}")

        return output_step_file

    finally:
        App.closeDocument(document.Name)


# ---------------------------------------------------------------------------
# Read parameters from environment variables
# ---------------------------------------------------------------------------

window_h_mm = read_float_environment_variable(
    "WINDOW_H_MM",
    17.2
)

window_w_mm = read_float_environment_variable(
    "WINDOW_W_MM",
    7.0
)

core_inner_diameter_mm = read_float_environment_variable(
    "CORE_INNER_DIAMETER_MM",
    13.45
)

flange_thickness_mm = read_float_environment_variable(
    "FLANGE_THICKNESS_MM",
    2.0
)

clearance = read_float_environment_variable(
    "CLEARANCE",
    0.3
)

inner_edge_radius_mm = read_float_environment_variable(
    "INNER_EDGE_RADIUS",
    0.6
)

outer_edge_radius_mm = read_float_environment_variable(
    "OUTER_EDGE_RADIUS",
    0.6
)

enable_wire_slots = read_bool_environment_variable(
    "ENABLE_WIRE_SLOTS",
    True
)

wire_slot_width_mm = read_float_environment_variable(
    "WIRE_SLOT_WIDTH",
    4.0
)

wire_slot_depth_mm = read_float_environment_variable(
    "WIRE_SLOT_DEPTH",
    window_w_mm - flange_thickness_mm
)

wire_slot_height_mm = read_float_environment_variable(
    "WIRE_SLOT_HEIGHT",
    flange_thickness_mm + 0.5
)

wire_slots_position = read_string_environment_variable(
    "WIRE_SLOTS_POSITION",
    "both"
)

output_step_file = os.environ.get(
    "OUTPUT_STEP_FILE",
    "./pq4040_round_bobbin.step"
)

save_freecad_file = os.environ.get(
    "SAVE_FCSTD_FILE",
    "0"
).strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Generate and export
# ---------------------------------------------------------------------------

try:
    logger.info("Generating PQ round bobbin with parameters:")
    logger.info(f"  window_h_mm: {window_h_mm}")
    logger.info(f"  window_w_mm: {window_w_mm}")
    logger.info(f"  core_inner_diameter_mm: {core_inner_diameter_mm}")
    logger.info(f"  flange_thickness_mm: {flange_thickness_mm}")
    logger.info(f"  clearance: {clearance}")
    logger.info(f"  inner_edge_radius_mm: {inner_edge_radius_mm}")
    logger.info(f"  outer_edge_radius_mm: {outer_edge_radius_mm}")
    logger.info(f"  enable_wire_slots: {enable_wire_slots}")
    logger.info(f"  wire_slots_position: {wire_slots_position}")
    logger.info(f"  wire_slot_width_mm: {wire_slot_width_mm}")
    logger.info(f"  wire_slot_depth_mm: {wire_slot_depth_mm}")
    logger.info(f"  wire_slot_height_mm: {wire_slot_height_mm}")
    logger.info(f"  output_step_file: {output_step_file}")

    export_round_bobbin_step(
        window_h_mm=window_h_mm,
        window_w_mm=window_w_mm,
        core_inner_diameter_mm=core_inner_diameter_mm,
        flange_thickness_mm=flange_thickness_mm,
        clearance=clearance,
        inner_edge_radius_mm=inner_edge_radius_mm,
        outer_edge_radius_mm=outer_edge_radius_mm,
        enable_wire_slots=enable_wire_slots,
        wire_slot_width_mm=wire_slot_width_mm,
        wire_slot_depth_mm=wire_slot_depth_mm,
        wire_slot_height_mm=wire_slot_height_mm,
        wire_slots_position=wire_slots_position,
        output_step_file=output_step_file,
        save_freecad_file=save_freecad_file
    )

except Exception as error:
    logger.exception("Failed to create PQ round bobbin: %s",error)
    print(f"ERROR while creating PQ40/40 round bobbin: {error}", file=sys.stderr)
    sys.exit(1)
