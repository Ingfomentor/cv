function plot_jaw_histograms(data_file, image_file)
  % load histogram data
  load(data_file);

  slices = size(histograms)(1);

  % create figure, plot histogram data and save to requested file
  fh = figure;
  hold on
  
  for slice = slices:-1:1
    histogram   = histograms(slice,:);
    split_upper = splits_upper(slice);
    split_lower = splits_lower(slice);
    top = max(histogram) * 1.1
    len = length(histogram)

    subplot(slices, 1, slices-slice+1)
    plot(histogram, 'lineWidth', 4);
    hold on
    plot([split_upper,split_upper],[0,top], 'r', 'lineWidth', 4)
    plot([split_lower,split_lower],[0,top], 'g', 'lineWidth', 4)
    axis([0 len 0 top])
    legend('summed intensities', 'upper split', 'lower split')
  endfor

  print(image_file, '-tight', '-color');
  close(fh);
end
