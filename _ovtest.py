import math, time
from AppKit import (NSApplication, NSWindow, NSView, NSColor, NSScreen,
                    NSWindowStyleMaskBorderless, NSBackingStoreBuffered,
                    NSScreenSaverWindowLevel, NSMakeRect,
                    NSWindowCollectionBehaviorCanJoinAllSpaces,
                    NSWindowCollectionBehaviorStationary,
                    NSWindowCollectionBehaviorFullScreenAuxiliary)
from Foundation import NSTimer
app = NSApplication.sharedApplication()
frame = NSScreen.mainScreen().frame()
win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
    frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
win.setOpaque_(False); win.setBackgroundColor_(NSColor.clearColor())
win.setIgnoresMouseEvents_(True); win.setLevel_(NSScreenSaverWindowLevel)
win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                           | NSWindowCollectionBehaviorStationary
                           | NSWindowCollectionBehaviorFullScreenAuxiliary)
root = NSView.alloc().initWithFrame_(frame); win.setContentView_(root)
dot = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 44, 44))
dot.setWantsLayer_(True)
dot.layer().setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(
    0.35, 0.65, 1.0, 0.55).CGColor())
dot.layer().setCornerRadius_(22.0); dot.layer().setBorderWidth_(2.0)
dot.layer().setBorderColor_(NSColor.whiteColor().CGColor())
root.addSubview_(dot); win.orderFrontRegardless()
t0 = time.time()
def tick(timer):
    e = time.time() - t0
    if e > 4.0: app.stop_(None); return
    dot.setFrameOrigin_((frame.size.width/2 + 380*math.cos(e*2.2) - 22,
                         frame.size.height/2 + 260*math.sin(e*2.2) - 22))
NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1/60.0, True, tick)
print("overlay up; dot orbiting for 4s over whatever is on screen")
app.run()
print("overlay closed cleanly")
