use crate::ImVec2;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum StyleVar {
    Alpha(f32),
    WindowPadding(ImVec2),
    WindowRounding(f32),
    WindowBorderSize(f32),
    WindowMinSize(ImVec2),
    ChildRounding(f32),
    ChildBorderSize(f32),
    PopupRounding(f32),
    PopupBorderSize(f32),
    FramePadding(ImVec2),
    FrameRounding(f32),
    FrameBorderSize(f32),
    ItemSpacing(ImVec2),
    ItemInnerSpacing(ImVec2),
    IndentSpacing(f32),
    GrabMinSize(f32),
    ButtonTextAlign(ImVec2),
}
