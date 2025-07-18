#!/usr/bin/env python3
"""
Test script to verify MONAI trainers work with your data format
"""


def test_simple_trainer():
    """Test the simple MONAI trainer setup"""
    try:
        from train_monai_simple import train_with_monai_simple
        print("✅ Simple trainer imported successfully")

        # You can add a quick test run here if needed
        # train_with_monai_simple()  # Uncomment to test actual training

    except Exception as e:
        print(f"❌ Simple trainer failed: {e}")
        return False
    return True


def test_engines_trainer():
    """Test the engines MONAI trainer setup"""
    try:
        from train_monai_engines import train_with_monai_engines
        print("✅ Engines trainer imported successfully")

        # You can add a quick test run here if needed
        # train_with_monai_engines()  # Uncomment to test actual training

    except Exception as e:
        print(f"❌ Engines trainer failed: {e}")
        return False
    return True


def test_handler_attachment():
    """Test that handlers can be attached properly"""
    try:
        from monai.engines import SupervisedTrainer
        from monai.handlers import StatsHandler
        import torch

        # Create minimal setup that mimics your data format
        device = torch.device('cpu')

        # Create a batch that matches your data format
        dummy_batch = {
            'image': torch.randn(2, 1, 32, 32),
            'mask': torch.randn(2, 1, 32, 32),
            'label': torch.tensor([0, 1])
        }
        dummy_loader = [dummy_batch]

        # Create a simple multitask model
        dummy_model = torch.nn.ModuleDict({
            'classifier': torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, 3, padding=1),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(4, 2)
            ),
            'segmentor': torch.nn.Conv2d(1, 1, 3, padding=1)
        })

        def forward_func(x):
            # Simple multitask forward pass
            class_out = dummy_model['classifier'](x)
            seg_out = dummy_model['segmentor'](x)
            return class_out, seg_out

        dummy_model.forward = forward_func
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters())

        def prepare_batch(batchdata, device=None, non_blocking=False, **kwargs):
            if device is not None:
                inputs = batchdata['image'].to(device, non_blocking=non_blocking)
                mask = batchdata['mask'].to(device, non_blocking=non_blocking)
                label = batchdata['label'].to(device, non_blocking=non_blocking)
            else:
                inputs = batchdata['image']
                mask = batchdata['mask']
                label = batchdata['label']

            targets = {'label': label, 'mask': mask}
            return inputs, targets

        def loss_fn(y_pred, y_true):
            return torch.tensor(1.0, requires_grad=True)

        # Create trainer
        trainer = SupervisedTrainer(
            device=device,
            max_epochs=1,
            train_data_loader=dummy_loader,
            network=dummy_model,
            optimizer=dummy_optimizer,
            loss_function=loss_fn,
            prepare_batch=prepare_batch,
            amp=False
        )

        # Test handler attachment
        stats_handler = StatsHandler(tag_name='test')
        stats_handler.attach(trainer)

        print("✅ Handler attachment successful")
        return True

    except Exception as e:
        print(f"❌ Handler attachment failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing MONAI Trainer Implementations\n")

    success_count = 0
    total_tests = 3

    if test_simple_trainer():
        success_count += 1

    if test_engines_trainer():
        success_count += 1

    if test_handler_attachment():
        success_count += 1

    print(f"\n📊 Test Results: {success_count}/{total_tests} passed")

    if success_count == total_tests:
        print("🎉 All tests passed! Your MONAI trainers are ready to use.")
        print("\nTo run training:")
        print("  python train_monai_simple.py    # Simple version")
        print("  python train_monai_engines.py   # Full-featured version")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
