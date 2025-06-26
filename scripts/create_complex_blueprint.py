#!/usr/bin/env python3
"""
Create Complex Blueprint with Many Doors

Creates a complex architectural blueprint with multiple doors
at different angles for testing detection methods.
"""

from PIL import Image, ImageDraw

def create_complex_blueprint():
    """Create a complex blueprint with many doors at various angles"""
    
    print('Creating complex blueprint...')
    
    # Create a complex blueprint with many doors at different angles
    width, height = 1000, 800
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Wall thickness
    wall_thickness = 6

    # Create a complex floor plan with multiple rooms
    # Outer walls
    draw.rectangle([30, 30, width-30, height-30], outline='black', width=wall_thickness)

    # Create internal walls to make a complex layout
    walls = [
        # Horizontal walls
        ([30, 200], [400, 200]),
        ([600, 200], [width-30, 200]),
        ([30, 400], [300, 400]),
        ([500, 400], [width-30, 400]),
        ([30, 600], [width-30, 600]),
        
        # Vertical walls  
        ([200, 30], [200, 200]),
        ([400, 200], [400, 400]),
        ([600, 30], [600, 600]),
        ([800, 200], [800, height-30]),
        ([300, 400], [300, 600]),
    ]

    for wall in walls:
        draw.line([wall[0][0], wall[0][1], wall[1][0], wall[1][1]], fill='black', width=wall_thickness)

    # Door parameters
    door_width = 60
    door_gap = 8

    # Create 15 doors at different positions and angles
    doors = [
        # Vertical wall doors (swing horizontally)
        {'pos': (200, 120), 'type': 'vertical'},
        {'pos': (400, 300), 'type': 'vertical'},
        {'pos': (600, 150), 'type': 'vertical'},
        {'pos': (600, 450), 'type': 'vertical'},
        {'pos': (800, 350), 'type': 'vertical'},
        {'pos': (300, 500), 'type': 'vertical'},
        {'pos': (30, 300), 'type': 'vertical', 'size': 'large'},  # entrance
        
        # Horizontal wall doors (swing vertically)
        {'pos': (150, 200), 'type': 'horizontal'},
        {'pos': (500, 200), 'type': 'horizontal'},
        {'pos': (150, 400), 'type': 'horizontal'},
        {'pos': (700, 400), 'type': 'horizontal'},
        {'pos': (400, 600), 'type': 'horizontal'},
        {'pos': (800, 600), 'type': 'horizontal'},
        {'pos': (500, 30), 'type': 'horizontal', 'size': 'large'},  # main entrance
        {'pos': (width-30, 500), 'type': 'vertical', 'size': 'large'},  # side entrance
    ]

    print(f'Drawing {len(doors)} doors...')

    # Draw doors
    for i, door in enumerate(doors):
        x, y = door['pos']
        door_type = door['type']
        size = door.get('size', 'normal')
        width_adj = int(door_width * 1.5) if size == 'large' else door_width
        
        if door_type == 'vertical':
            # Door in vertical wall (opens horizontally)
            # Create gap in wall
            gap_rect = [x-door_gap//2, y-width_adj//2, x+door_gap//2, y+width_adj//2]
            draw.rectangle(gap_rect, fill='white', outline='white')
            
            # Draw door swing arc (quarter circle)
            arc_radius = width_adj // 2
            arc_box = [x, y-arc_radius, x+arc_radius, y+arc_radius]
            draw.arc(arc_box, start=0, end=90, fill='gray', width=2)
            
        else:
            # Door in horizontal wall (opens vertically)
            # Create gap in wall
            gap_rect = [x-width_adj//2, y-door_gap//2, x+width_adj//2, y+door_gap//2]
            draw.rectangle(gap_rect, fill='white', outline='white')
            
            # Draw door swing arc (quarter circle)
            arc_radius = width_adj // 2
            arc_box = [x-arc_radius, y, x+arc_radius, y+arc_radius]
            draw.arc(arc_box, start=180, end=270, fill='gray', width=2)

    print('Adding room labels...')

    # Add room labels (no door labels!)
    room_labels = [
        (115, 115, 'Living Room'),
        (300, 115, 'Kitchen'), 
        (700, 115, 'Bedroom 1'),
        (115, 300, 'Dining'),
        (250, 300, 'Bath'),
        (700, 300, 'Bedroom 2'),
        (115, 500, 'Storage'),
        (500, 500, 'Office'),
        (850, 350, 'Closet'),
    ]

    for x, y, label in room_labels:
        draw.text((x, y), label, fill='black')

    # Save the complex blueprint
    output_path = 'images/blueprints/blueprint_with_doors.png'
    img.save(output_path)
    print(f'✅ Created {output_path} with {len(doors)} doors at various angles!')
    
    return output_path, len(doors)

if __name__ == "__main__":
    create_complex_blueprint() 