import argparse
import time
from pathlib import Path

from pipeline import PipelineConfig, HandDetectionPipeline


def main():
    """Main execution function"""
    # Get script directory
    script_dir = Path(__file__)
    
    # Configure pipeline
    model = "mediapipe"
    multi_run = False

    if multi_run:
        cameras = {
            "camera01": 0.5,
            "camera02": 0.4,
            "camera03": 0.4,
            "camera04": 0.4,
            "camera05": 0.4,
            "camera06": 0.4
        }
        min_tracking_confidence = 0.75
        run_start = time.time()
        for camera, conf in cameras.items():
            orbbec_cam = True if camera not in ['camera05', 'camera06'] else False

            print(camera)
            if camera in ['camera01', 'camera05', 'camera06']:
                continue
            
            # Build paths relative to script directory
            base_path = script_dir.parent.parent.parent / "data" / "input"
            output_path = script_dir.parent.parent / "output" / "test" /  model / "diff_confs_run" / camera / f"conf_{conf:.2f}"

            config = PipelineConfig(
                input_path=str(base_path / "orbbec" / camera),
                camera_path=str(base_path),
                output_path=str(output_path),
                camera_name=camera,
                orbbec_cam=orbbec_cam,
                model=model,
                min_detection_confidence=conf,
                min_tracking_confidence=min_tracking_confidence,
                crop_size=256,
                bbox_shift=150,
                max_workers=10,     # Number of threads
                batch_size=20       # Images per batch
            )
            
            # Create pipeline
            pipeline = HandDetectionPipeline(config)

            start = time.time()
            
            # Use single-threaded version
            pipeline.run(start_idx=0, end_idx=0)
            
            end = time.time()
            
            elapsed_time = end - start
            pipeline.logger.info(f"Pipeline execution time: {elapsed_time:.2f} seconds")

        run_end = time.time()
        run_time = run_end - run_start
        print(f"Entire run execution time: {run_time:.2f} seconds")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cam', type=int, default=1)
        parser.add_argument('--start', type=int, default=0)
        parser.add_argument('--end', type=int, default=20)
        parser.add_argument('--verbose', type=bool, default=False)
        args = parser.parse_args()

        camera = f"camera0{args.cam}"
        orbbec_cam = True if camera not in ['camera05', 'camera06'] else False
        conf = 0.35
        min_tracking_confidence = 0.6
        print(camera)

        subdir_name = "20250519_Testing"
        # Build paths relative to script directory
        base_path = script_dir.parent.parent / "data" / subdir_name
        output_path = script_dir.parent.parent / "output" / subdir_name / f"{model}_{conf:.2f}" / camera 

        config = PipelineConfig(
            input_path=str(base_path / camera),
            camera_path=str(base_path),
            output_path=str(output_path),
            verbose=args.verbose,
            camera_name=camera,
            orbbec_cam=orbbec_cam,
            model=model,
            min_detection_confidence=conf,
            min_tracking_confidence=min_tracking_confidence,
            crop_size=256,
            bbox_shift=150,
            max_workers=10,     # Number of threads
            batch_size=20       # Images per batch
        )
        
        # Create pipeline
        pipeline = HandDetectionPipeline(config)

        start = time.time()
        
        # Use multithreaded version (uncomment to enable)
        # pipeline.run_multithreaded(start_idx=args.start, end_idx=args.end)
        
        # Use single-threaded version
        pipeline.run(start_idx=args.start, end_idx=args.end)
        
        end = time.time()
        
        elapsed_time = end - start
        pipeline.logger.info(f"Pipeline execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()