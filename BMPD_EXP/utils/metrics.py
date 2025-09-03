import torch
import numpy


class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    # def get_metrics(self):
    #     h = self.mat.float()
    #     acc = torch.diag(h).sum() / h.sum()
    #     iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
    #     return torch.mean(iu[1:]).item(), acc.item()

    def get_metrics_test(self):
        h = self.mat.float()
        all_acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        acc = torch.diag(h) / h.sum(1)
        precision = torch.diag(h) / h.sum(0)
        recall = torch.diag(h) / h.sum(1)

        # Calculate F1 score using macro-average
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        return torch.mean(iu[1:]).item(), torch.mean(acc[1:]).item(), torch.mean(f1[1:]), iu, acc, f1


class MIoU(object):
    def __init__(self, num_classes, ignore_index):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.inter, self.union = 0, 0
        self.correct, self.label = 0, 0
        self.iou = numpy.array([0 for _ in range(num_classes)])
        self.acc = 0.0

    def get_metric_results(self):
        return numpy.round(self.iou, 4), numpy.round(self.acc, 4)
        # if class_list is None:
        #     return numpy.round(self.iou.mean().item(), 4), \
        #         numpy.round(self.acc, 4)
        # else:
        #     return numpy.round(self.iou[class_list].mean().item(), 4), \
        #         numpy.round(self.acc, 4)

    def __call__(self, x, y):
        curr_correct, curr_label, curr_inter, curr_union = self.calculate_current_sample(x, y)
        # calculates the overall miou and acc
        self.correct = self.correct + curr_correct
        self.label = self.label + curr_label
        self.inter = self.inter + curr_inter
        self.union = self.union + curr_union

        self.acc = 1.0 * self.correct / (numpy.spacing(1) + self.label)
        self.iou = 1.0 * self.inter / (numpy.spacing(1) + self.union)
        return self.get_metric_results()

    def calculate_current_sample(self, output, target):
        # output => BxCxHxW (logits)
        # target => Bx1xHxW
        target[target == self.ignore_index] = -1
        correct, labeled = self.batch_pix_accuracy(output.data, target)
        inter, union = self.batch_intersection_union(output.data, target, self.num_classes)
        return [numpy.round(correct, 5), numpy.round(labeled, 5), numpy.round(inter, 5), numpy.round(union, 5)]

    @ staticmethod
    def batch_pix_accuracy(output, target):
        _, predict = torch.max(output, 1)

        predict = predict.int() + 1
        target = target.int() + 1

        pixel_labeled = (target > 0).sum()
        pixel_correct = ((predict == target) * (target > 0)).sum()
        assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
        return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

    @ staticmethod
    def batch_intersection_union(output, target, num_class):
        _, predict = torch.max(output, 1)
        predict = predict + 1
        target = target + 1

        predict = predict * (target > 0).long()
        intersection = predict * (predict == target).long()

        area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
        area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
        area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
        return area_inter.cpu().numpy(), area_union.cpu().numpy()