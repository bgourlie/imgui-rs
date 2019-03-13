#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("test_thing", log::LevelFilter::Trace)
        .init();

    let config: Config = Default::default();
    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}
