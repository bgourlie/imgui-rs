use parking_lot::ReentrantMutex;
use std::borrow::Cow;
use std::cell::RefCell;
use std::ffi::CStr;
use std::ops::Drop;
use std::ptr;
use std::rc::Rc;

use crate::font_atlas::{FontAtlas, FontAtlasRefMut, SharedFontAtlas};
use crate::io::Io;
use crate::string::ImStr;
use crate::style::Style;
use crate::sys;

/// An imgui-rs context.
///
/// A context needs to be created to access most library functions. Due to current Dear ImGui
/// design choices, at most one active Context can exist at any time. This limitation will likely
/// be removed in a future Dear ImGui version.
///
/// If you need more than one context, you can use suspended contexts. As long as only one context
/// is active at a time, it's possible to have multiple independent contexts.
///
/// # Examples
///
/// Creating a new active context:
/// ```
/// let ctx = imgui::Context::create();
/// // ctx is dropped naturally when it goes out of scope, which deactivates and destroys the
/// // context
/// ```
///
/// Never try to create an active context when another one is active:
///
/// ```should_panic
/// let ctx1 = imgui::Context::create();
///
/// let ctx2 = imgui::Context::create(); // PANIC
/// ```
///
/// Suspending an active context allows you to create another active context:
///
/// ```
/// let ctx1 = imgui::Context::create();
/// let suspended1 = ctx1.suspend();
/// let ctx2 = imgui::Context::create(); // this is now OK
/// ```
#[derive(Debug)]
pub struct Context {
    raw: *mut sys::ImGuiContext,
    shared_font_atlas: Option<Rc<RefCell<SharedFontAtlas>>>,
    ini_filename: Option<Cow<'static, ImStr>>,
    log_filename: Option<Cow<'static, ImStr>>,
}

lazy_static! {
    // This mutex needs to be used to guard all public functions that can affect the underlying
    // Dear ImGui active context
    static ref CTX_MUTEX: ReentrantMutex<()> = ReentrantMutex::new(());
}

fn clear_current_context() {
    unsafe {
        sys::igSetCurrentContext(ptr::null_mut());
    }
}
fn no_current_context() -> bool {
    let ctx = unsafe { sys::igGetCurrentContext() };
    ctx.is_null()
}

impl Context {
    /// Creates a new active imgui-rs context.
    ///
    /// # Panics
    ///
    /// Panics if an active context already exists
    pub fn create() -> Self {
        Self::create_internal(None)
    }
    /// Creates a new active imgui-rs context with a shared font atlas.
    ///
    /// # Panics
    ///
    /// Panics if an active context already exists
    pub fn create_with_shared_font_atlas(shared_font_atlas: Rc<RefCell<SharedFontAtlas>>) -> Self {
        Self::create_internal(Some(shared_font_atlas))
    }
    /// Suspends this context so another context can be the active context.
    pub fn suspend(self) -> SuspendedContext {
        let _guard = CTX_MUTEX.lock();
        assert!(
            self.is_current_context(),
            "context to be suspended is not the active context"
        );
        clear_current_context();
        SuspendedContext(self)
    }
    pub fn ini_filename(&self) -> Option<&ImStr> {
        let io = self.io();
        if io.ini_filename.is_null() {
            None
        } else {
            unsafe { Some(ImStr::from_cstr_unchecked(CStr::from_ptr(io.ini_filename))) }
        }
    }
    pub fn set_ini_filename<T: Into<Option<Cow<'static, ImStr>>>>(&mut self, ini_filename: T) {
        let value = ini_filename.into();
        {
            let io = self.io_mut();
            io.ini_filename = match value {
                Some(ref x) => x.as_ptr(),
                None => ptr::null(),
            }
        }
        self.ini_filename = value;
    }
    pub fn log_filename(&self) -> Option<&ImStr> {
        let io = self.io();
        if io.log_filename.is_null() {
            None
        } else {
            unsafe { Some(ImStr::from_cstr_unchecked(CStr::from_ptr(io.log_filename))) }
        }
    }
    pub fn set_log_filename<T: Into<Option<Cow<'static, ImStr>>>>(&mut self, log_filename: T) {
        let value = log_filename.into();
        {
            let io = self.io_mut();
            io.log_filename = match value {
                Some(ref x) => x.as_ptr(),
                None => ptr::null(),
            }
        }
        self.log_filename = value;
    }
    pub fn load_ini_settings(&mut self, data: &str) {
        unsafe {
            sys::igLoadIniSettingsFromMemory(data.as_ptr() as *const _, data.len())
        }
    }
    pub fn save_ini_settings(&mut self, buf: &mut String) {
        let data = unsafe { CStr::from_ptr(sys::igSaveIniSettingsToMemory(ptr::null_mut())) };
        buf.push_str(&data.to_string_lossy());
    }
    fn create_internal(shared_font_atlas: Option<Rc<RefCell<SharedFontAtlas>>>) -> Self {
        let _guard = CTX_MUTEX.lock();
        assert!(
            no_current_context(),
            "A new active context cannot be created, because another one already exists"
        );
        // Dear ImGui implicitly sets the current context during igCreateContext if the current
        // context doesn't exists
        let raw = unsafe { sys::igCreateContext(ptr::null_mut()) };
        Context {
            raw,
            shared_font_atlas,
            ini_filename: None,
            log_filename: None,
        }
    }
    fn is_current_context(&self) -> bool {
        let ctx = unsafe { sys::igGetCurrentContext() };
        self.raw == ctx
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let _guard = CTX_MUTEX.lock();
        // If this context is the active context, Dear ImGui automatically deactivates it during
        // destruction
        unsafe {
            sys::igDestroyContext(self.raw);
        }
    }
}

/// A suspended imgui-rs context.
///
/// A suspended context retains its state, but is not usable without activating it first.
///
/// # Examples
///
/// Suspended contexts are not directly very useful, but you can activate them:
///
/// ```
/// let suspended = imgui::SuspendedContext::create();
/// match suspended.activate() {
///   Ok(ctx) => {
///     // ctx is now the active context
///   },
///   Err(suspended) => {
///     // activation failed, so you get the suspended context back
///   }
/// }
/// ```
#[derive(Debug)]
pub struct SuspendedContext(Context);

impl SuspendedContext {
    /// Creates a new suspended imgui-rs context.
    pub fn create() -> Self {
        Self::create_internal(None)
    }
    /// Creates a new suspended imgui-rs context with a shared font atlas.
    pub fn create_with_shared_font_atlas(shared_font_atlas: Rc<RefCell<SharedFontAtlas>>) -> Self {
        Self::create_internal(Some(shared_font_atlas))
    }
    /// Attempts to activate this suspended context.
    ///
    /// If there is no active context, this suspended context is activated and `Ok` is returned,
    /// containing the activated context.
    /// If there is already an active context, nothing happens and `Err` is returned, containing
    /// the original suspended context.
    pub fn activate(self) -> Result<Context, SuspendedContext> {
        let _guard = CTX_MUTEX.lock();
        if no_current_context() {
            unsafe {
                sys::igSetCurrentContext(self.0.raw);
            }
            Ok(self.0)
        } else {
            Err(self)
        }
    }
    fn create_internal(shared_font_atlas: Option<Rc<RefCell<SharedFontAtlas>>>) -> Self {
        let _guard = CTX_MUTEX.lock();
        let raw = unsafe { sys::igCreateContext(ptr::null_mut()) };
        let ctx = Context {
            raw,
            shared_font_atlas,
            ini_filename: None,
            log_filename: None,
        };
        if ctx.is_current_context() {
            // Oops, the context was activated -> deactivate
            clear_current_context();
        }
        SuspendedContext(ctx)
    }
}

#[test]
fn test_one_context() {
    let _guard = crate::test::TEST_MUTEX.lock();
    let _ctx = Context::create();
    assert!(!no_current_context());
}

#[test]
fn test_drop_clears_current_context() {
    let _guard = crate::test::TEST_MUTEX.lock();
    {
        let _ctx1 = Context::create();
        assert!(!no_current_context());
    }
    assert!(no_current_context());
    {
        let _ctx2 = Context::create();
        assert!(!no_current_context());
    }
    assert!(no_current_context());
}

#[test]
fn test_new_suspended() {
    let _guard = crate::test::TEST_MUTEX.lock();
    let ctx = Context::create();
    let _suspended = SuspendedContext::create();
    assert!(ctx.is_current_context());
    ::std::mem::drop(_suspended);
    assert!(ctx.is_current_context());
}

#[test]
fn test_suspend() {
    let _guard = crate::test::TEST_MUTEX.lock();
    let ctx = Context::create();
    assert!(!no_current_context());
    let _suspended = ctx.suspend();
    assert!(no_current_context());
    let _ctx2 = Context::create();
}

#[test]
fn test_drop_suspended() {
    let _guard = crate::test::TEST_MUTEX.lock();
    let suspended = Context::create().suspend();
    assert!(no_current_context());
    let ctx2 = Context::create();
    ::std::mem::drop(suspended);
    assert!(ctx2.is_current_context());
}

#[test]
fn test_suspend_activate() {
    let _guard = crate::test::TEST_MUTEX.lock();
    let suspended = Context::create().suspend();
    assert!(no_current_context());
    let ctx = suspended.activate().unwrap();
    assert!(ctx.is_current_context());
}

#[test]
fn test_suspend_failure() {
    let _guard = crate::test::TEST_MUTEX.lock();
    let suspended = Context::create().suspend();
    let _ctx = Context::create();
    assert!(suspended.activate().is_err());
}

#[test]
fn test_shared_font_atlas() {
    let _guard = crate::test::TEST_MUTEX.lock();
    let atlas = Rc::new(RefCell::new(SharedFontAtlas::create()));
    let suspended1 = SuspendedContext::create_with_shared_font_atlas(atlas.clone());
    let mut ctx2 = Context::create_with_shared_font_atlas(atlas.clone());
    {
        let _borrow = ctx2.fonts();
    }
    let _suspended2 = ctx2.suspend();
    let mut ctx = suspended1.activate().unwrap();
    let _borrow = ctx.fonts();
}

#[test]
#[should_panic]
fn test_shared_font_atlas_borrow_panic() {
    let _guard = crate::test::TEST_MUTEX.lock();
    let atlas = Rc::new(RefCell::new(SharedFontAtlas::create()));
    let _suspended = SuspendedContext::create_with_shared_font_atlas(atlas.clone());
    let mut ctx = Context::create_with_shared_font_atlas(atlas.clone());
    let _borrow1 = atlas.borrow();
    let _borrow2 = ctx.fonts();
}

#[test]
fn test_ini_load_save() {
    let (_guard, mut ctx) = crate::test::test_ctx();
    let data = "[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0";
    ctx.load_ini_settings(&data);
    let mut buf = String::new();
    ctx.save_ini_settings(&mut buf);
    assert_eq!(data.trim(), buf.trim());
}

impl Context {
    /// Returns an immutable reference to the inputs/outputs object
    pub fn io(&self) -> &Io {
        unsafe {
            // safe because Io is a transparent wrapper around sys::ImGuiIO
            &*(sys::igGetIO() as *const Io)
        }
    }
    /// Returns a mutable reference to the inputs/outputs object
    pub fn io_mut(&mut self) -> &mut Io {
        unsafe {
            // safe because Io is a transparent wrapper around sys::ImGuiIO
            &mut *(sys::igGetIO() as *mut Io)
        }
    }
    /// Returns an immutable reference to the user interface style
    pub fn style(&self) -> &Style {
        unsafe {
            // safe because Style is a transparent wrapper around sys::ImGuiStyle
            &*(sys::igGetStyle() as *const Style)
        }
    }
    /// Returns a mutable reference to the user interface style
    pub fn style_mut(&mut self) -> &mut Style {
        unsafe {
            // safe because Style is a transparent wrapper around sys::ImGuiStyle
            &mut *(sys::igGetStyle() as *mut Style)
        }
    }
    /// Returns a mutable reference to the font atlas.
    ///
    /// # Panics
    ///
    /// Panics if the context uses a shared font atlas that is already borrowed
    pub fn fonts(&mut self) -> FontAtlasRefMut {
        match self.shared_font_atlas {
            Some(ref font_atlas) => FontAtlasRefMut::Shared(font_atlas.borrow_mut()),
            None => unsafe {
                let io = sys::igGetIO();
                // safe because FontAtlas is a transparent wrapper around sys::ImFontAtlas
                let fonts = &mut *((*io).Fonts as *mut FontAtlas);
                FontAtlasRefMut::Unique(fonts)
            },
        }
    }
}
