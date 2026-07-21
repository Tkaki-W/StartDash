GRBL = StageControl(PORT, baudrate=BAUD, stage_type='GRBL')
pos, t = GRBL.stage.get_current_pos()  # StageControlに get_current_pos がないので stage経由
x, y, z = pos
print(f'x={x}, y={y}, z={z}')