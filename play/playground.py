import pygame as pg
import gymnasium as gym
import arcle
from arcle.loaders import ARCLoader

env = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=None, data_loader=ARCLoader(), max_grid_size=(30,30), colors = 10)

pg.init()



# sizes
W=1280
H=720
gridmargin = int(50*W/1920)
gridsize =  (W-4*gridmargin)/2.5
toolfontsize = int(30*W/1920)
palettesize = gridsize/10
objbtnsize = palettesize*1.25

# texts
descfont = pg.font.SysFont('Cascadia Code' , toolfontsize)
colortool_text = descfont.render('Color Fill',True,(255,255,255))
floodtool_text = descfont.render('Flood Fill',True,(255,255,255))

# colors
dircolor = (255,205,245)
rotcolor = (223,255,200)
flipcolor = (205,245,255)
clipcolor = (255,200, 195)
cmap = [(int(st[0:2],16), int(st[2:4],16), int(st[4:6],16)) for st in ['000000', '0074D9', 'FF4136', '2ECC40', 'FFDC00', 'AAAAAA', 'F012BE', 'FF851B', '7FCBFF', '870C25']]

# states
colorset = 0
rgh, rgw = 0,0


screen = pg.display.set_mode((W,H))
clock = pg.time.Clock()
running = True
state, info= env.reset()



btn_border = 30
while running:
    screen.fill((0,0,0))
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    
    input_ = state['input']
    ih, iw = state['input_dim']

    grid = state['grid']
    gh, gw = state['grid_dim']
    rgh, rgw = gh, gw

    clip = state['clip']
    ch, cw = state['clip_dim']
    
    selected = state['selected']

    

    # input grid
    pixsize = gridsize/max(ih, iw)
    for i in range(ih):
        for j in range(iw):
            pg.draw.rect(screen,cmap[input_[i,j]], pg.Rect(gridmargin+pixsize*i , gridmargin+pixsize*j,pixsize,pixsize))
            pg.draw.rect(screen,(128,128,128), pg.Rect(gridmargin+pixsize*i , gridmargin-0.5+pixsize*j-0.5,pixsize+1,pixsize+1), 1)

    pixsize = gridsize/max(gh, gw)
    for i in range(gh):
        for j in range(gw):
            pg.draw.rect(screen,cmap[grid[i,j]], pg.Rect(2*gridmargin+gridsize+pixsize*i , gridmargin+pixsize*j,pixsize,pixsize))
            pg.draw.rect(screen,(128,128,128), pg.Rect(2*gridmargin+gridsize+pixsize*i , gridmargin-0.5+pixsize*j-0.5,pixsize+1,pixsize+1), 1)

    
    if ch!=0 and cw!=0:
        pixsize = gridsize/max(ch, cw)*0.5
        for i in range(ch):
            for j in range(cw):
                pg.draw.rect(screen,cmap[clip[i,j]], pg.Rect(3*gridmargin+2*gridsize+pixsize*i , gridmargin+gridsize/2+pixsize*j,pixsize,pixsize))
                pg.draw.rect(screen,(128,128,128), pg.Rect(3*gridmargin+2*gridsize+pixsize*i , gridmargin+gridsize/2-0.5+pixsize*j-0.5,pixsize+1,pixsize+1), 1)
    
    screen.blit(colortool_text,((gridmargin, gridmargin*2.5+ gridsize-toolfontsize)))
    screen.blit(floodtool_text,((gridmargin, gridmargin*4+palettesize+ gridsize-toolfontsize)))
    for i in range(10):
        pg.draw.rect(screen, color = cmap[i], rect = (gridmargin+palettesize*i, gridmargin*2.5+ gridsize, palettesize,palettesize))
        pg.draw.rect(screen, color = (255,255,255), rect = (gridmargin+palettesize*i-1, gridmargin*2.5 + gridsize-1, palettesize+2,palettesize+2), width=2)
        pg.draw.rect(screen, color = cmap[i], rect = (gridmargin+palettesize*i, gridmargin*2.5+2*palettesize+ gridsize, palettesize,palettesize))
        pg.draw.rect(screen, color = (255,255,255), rect = (gridmargin+palettesize*i-1, gridmargin*2.5 +2*palettesize+ gridsize-1, palettesize+2,palettesize+2), width=2)

    
    
    fliphbtn = pg.draw.rect(screen, color = flipcolor, rect = (gridsize*1.5+gridmargin*(-3)-objbtnsize/2,gridsize+gridmargin*2.5, objbtnsize, objbtnsize))
    flipvbtn = pg.draw.rect(screen, color = flipcolor, rect = (gridsize*1.5+gridmargin*(-3)-objbtnsize/2,gridsize+gridmargin*2.5+palettesize*1.75, objbtnsize, objbtnsize))

    ccwbtn = pg.draw.rect(screen,color = rotcolor, rect = ((gridsize*1.5+gridmargin*(-0.5)-objbtnsize/2,gridsize+gridmargin*2.5, objbtnsize, objbtnsize)))
    cwbtn = pg.draw.rect(screen,color = rotcolor, rect = ((gridsize*1.5+gridmargin*4.5-objbtnsize/2,gridsize+gridmargin*2.5, objbtnsize, objbtnsize)))
    pg.transform.rotate(screen,10)
    
    moveleftbtn = pg.draw.rect(screen, color = dircolor , rect = (gridsize*1.5+gridmargin*(-0.5)-objbtnsize/2,gridsize+gridmargin*2.5+palettesize*1.75, objbtnsize, objbtnsize))
    moveupbtn = pg.draw.rect(screen,color = dircolor, rect = ((gridsize*1.5+gridmargin*2-objbtnsize/2,gridsize+gridmargin*2.5, objbtnsize, objbtnsize)))
    movedownbtn = pg.draw.rect(screen, color = dircolor, rect = (gridsize*1.5+gridmargin*2-objbtnsize/2,gridsize+gridmargin*2.5+palettesize*1.75, objbtnsize, objbtnsize))
    moverightbtn = pg.draw.rect(screen, color = dircolor, rect = (gridsize*1.5+gridmargin*4.5-objbtnsize/2,gridsize+gridmargin*2.5+palettesize*1.75, objbtnsize, objbtnsize))

    copybtn = pg.draw.rect(screen, color = clipcolor, rect = (gridsize*1.5+gridmargin*7-objbtnsize/2,gridsize+gridmargin*2.5, objbtnsize, objbtnsize))
    pastebtn = pg.draw.rect(screen, color = clipcolor, rect = (gridsize*1.5+gridmargin*7-objbtnsize/2,gridsize+gridmargin*2.5+palettesize*1.75, objbtnsize, objbtnsize))

    
    copyfrominput = pg.draw.rect(screen, color = (255,255, 255), rect = (gridmargin+gridsize+gridmargin*0.125,gridsize/2-gridmargin*0.5,gridmargin*0.75,gridmargin*3), border_radius=btn_border)
    resetbtn = pg.draw.rect(screen, color = (255,255,255), rect = (gridmargin+gridsize+gridmargin*0.3,gridsize/2-gridmargin*5,gridmargin*0.75,gridmargin*3))
    inputgridarea = pg.draw.rect(screen, color = (255,255,255), rect=(gridmargin,gridmargin, gridsize,gridsize), width=5)
    editgridarea = pg.draw.rect(screen, color = (255,255,255), rect=(2*gridmargin+gridsize,gridmargin, gridsize,gridsize), width=5)
    pg.draw.rect(screen, color = (255,255,255), rect=(3*gridmargin+2*gridsize,gridmargin+gridsize/2, gridsize//2,gridsize//2), width=5)
    
    pg.display.flip()
    clock.tick(10)
    
    
env.close()
pg.quit()
